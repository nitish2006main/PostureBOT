/*
╔══════════════════════════════════════════════════════════════════════════════╗
║  PostureBOT_v4.ino  —  PostureBOT                                           ║
║  Target : ESP32-S3  (dual-core, FreeRTOS)                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OVERVIEW                                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Camera is physically mounted on the PostureBOT head unit.  Python (MediaPipe)║
║  sends the pixel error (how far the nose is from the screen centre) via     ║
║  serial.  The ESP32 runs a PID controller and drives the servos to chase    ║
║  the face.  XYZ coordinates are computed here from servo angles + face Z.  ║
║                                                                              ║
║  COORDINATE ORIGIN = calibration pose (face centred in camera frame)        ║
║   +X = face moved RIGHT    (pan increased from calib)                       ║
║   +Y = face moved UP       (tilt increased from calib)                      ║
║   +Z = face moved FURTHER  (apparent face size decreased)                   ║
║                                                                              ║
║  SERIAL PROTOCOL                                                             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   Python → ESP32:                                                            ║
║     "ERR,px,py"   Raw pixel error; ESP32 PID drives servos                  ║
║     "DIST,z_cm"   Current face Z in cm; used for XYZ computation            ║
║     "CALIB_OK"    Python detected stable face lock → save calib ticks       ║
║     "WARN"        Face/eyes missing, in grace window                        ║
║     "FOUND"       Face/eyes re-acquired                                     ║
║     "LOST"        Face missing ≥ 5 s  → auto-stop                          ║
║     "NOEYES"      Eyes closed ≥ 5 s  → auto-stop                           ║
║     "TILTL"       Yaw left    → OLED "Turn Right"                           ║
║     "TILTR"       Yaw right   → OLED "Turn Left"                            ║
║     "PITCHUP"     Pitch up    → OLED "Tilt Down"                            ║
║     "PITCHDOWN"   Pitch down  → OLED "Tilt Up"                              ║
║     "TILTOK"      All pose in range → clear message                         ║
║   ESP32 → Python:                                                            ║
║     "CALIBRATING"  Calibration mode started, Python should begin tracking   ║
║     "CALIB_DONE"   Calibration ticks saved                                  ║
║     "XYZ,x,y,z"   Live position in cm (computed from servo + DIST)          ║
║     "START"        Session started                                           ║
║     "PAUSE"        Session paused                                            ║
║     "RESUME"       Session resumed                                           ║
║     "STOP"         Session ended                                             ║
║                                                                              ║
║  FREERTOS TASK MAP                                                           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║   Core 1 (latency-sensitive)                                                ║
║     taskSerial  pri 4   USB serial send / receive with Python               ║
║     taskServo   pri 4   PID loop — moves servos to chase the nose           ║
║   Core 0 (background / IO)                                                  ║
║     taskBuzzer  pri 2   Warning audio feedback                              ║
║     taskOLED    pri 2   Live OLED refresh every 100 ms                     ║
║     taskMenu    pri 1   Encoder input + app state machine                   ║
║                                                                              ║
║  REQUIRED LIBRARIES (Arduino Library Manager)                               ║
║    Adafruit SSD1351  ·  Adafruit GFX  ·  Adafruit PWMServoDriver           ║
║    Preferences  (built-in with ESP32 Arduino core)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
*/

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <math.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1351.h>
#include <Adafruit_PWMServoDriver.h>
#include <Preferences.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_task_wdt.h"


// ═══════════════════════════════════════════════════════════════════════════════
//  PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════════
#define OLED_MOSI   10
#define OLED_CLK    11
#define OLED_CS      9
#define OLED_DC     15
#define OLED_RST     3

#define BUZZER_PIN   6
#define ENC_CLK     14
#define ENC_DT      13
#define ENC_SW      12


// ═══════════════════════════════════════════════════════════════════════════════
//  HARDWARE OBJECTS
// ═══════════════════════════════════════════════════════════════════════════════
#define SCREEN_W   128
#define SCREEN_H   128

Adafruit_SSD1351 tft(SCREEN_W, SCREEN_H, &SPI, OLED_CS, OLED_DC, OLED_RST);
Adafruit_PWMServoDriver pwm(0x40);
Preferences prefs;


// ═══════════════════════════════════════════════════════════════════════════════
//  PCA9685 SERVO CONFIG
// ═══════════════════════════════════════════════════════════════════════════════
#define SERVO_PAN_CH    15
#define SERVO_TILT_CH   14

// PAN servo tick map (left/right rotation):
//   380 = straight ahead — this is our "origin" (calibration centre)
//   160 = rotated far LEFT  (from the user's perspective)
//   600 = rotated far RIGHT (from the user's perspective)
//
// TILT servo tick map (up/down rotation):
//   355 = level (camera points straight forward) — origin
//   585 = pointing UP — ABSOLUTE MAXIMUM, do NOT exceed or the servo will strip
//   190 = pointing DOWN minimum
#define PAN_ORIGIN      380
#define PAN_MIN         160
#define PAN_MAX         600

#define TILT_ORIGIN     355
#define TILT_MIN        190
#define TILT_MAX        585

// Conversion factor: how many degrees does the camera move per one tick step?
// Used only for displaying angles on the OLED — not used by the PID maths directly.
#define PAN_DEG_PER_TICK   (180.0f / 450.0f)
#define TILT_DEG_PER_TICK  (180.0f / 450.0f)

// ── P gain ───────────────────────────────────────────────────────────────────
// Scales pixel error to a tick delta per update.  Raise for faster response;
// lower if the servo oscillates.
#define PAN_KP   0.07f
#define TILT_KP  0.07f

// Maximum ticks either axis can travel per Python update.
// Caps overshoot while still allowing fluid motion via the for-loop sweep.
#define MAX_STEP_TICKS  12

// Deadzone: ignore pixel errors smaller than ±15 pixels.
// This stops the camera from jittering when the nose is already very close to centre.
#define PID_DEADZONE_PX     15


// ── Safe servo write helpers ──────────────────────────────────────────────────
// Always go through these functions instead of calling pwm.setPWM() directly.
// They clamp the tick value to the safe range so we can never accidentally
// command a position that would damage the hardware.

void setPanTick(int tick) {
    if (tick < PAN_MIN)  tick = PAN_MIN;
    if (tick > PAN_MAX)  tick = PAN_MAX;
    pwm.setPWM(SERVO_PAN_CH, 0, tick);
}

void setTiltTick(int tick) {
    if (tick < TILT_MIN)  tick = TILT_MIN;
    if (tick > TILT_MAX)  tick = TILT_MAX;
    pwm.setPWM(SERVO_TILT_CH, 0, tick);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  BUZZER CONFIG
//  The buzzer is driven with a PWM tone using the ESP32's LEDC peripheral.
// ═══════════════════════════════════════════════════════════════════════════════
#define BUZZ_FREQ_HZ    1200    // base tone frequency in Hz (roughly a D note)
#define BUZZ_RESOLUTION    8    // 8-bit resolution → duty cycle range 0-255


// ═══════════════════════════════════════════════════════════════════════════════
//  OLED COLOURS  (RGB565 format)
//  The SSD1351 uses 16-bit colour: 5 bits red, 6 bits green, 5 bits blue.
//  These are pre-computed hex constants for the colours we use in the UI.
// ═══════════════════════════════════════════════════════════════════════════════
#define BLACK       0x0000
#define WHITE       0xFFFF
#define CYAN        0x07FF
#define MAGENTA     0xF81F
#define YELLOW      0xFFE0
#define GREEN       0x07E0
#define RED         0xF800
#define BLUE        0x001F
#define ORANGE      0xFD20
#define DARK_CYAN   0x03EF
#define GREY        0x39E7
#define DARK_GREY   0x18C3
#define LIME        0x87E0


// ═══════════════════════════════════════════════════════════════════════════════
//  FREERTOS SYNC OBJECTS
//  These let different tasks communicate and share hardware safely.
// ═══════════════════════════════════════════════════════════════════════════════

// xyQueue: a small FIFO buffer of pixel-error messages.
// taskSerial puts messages in; taskServo takes them out.
// Using a queue means taskServo never reads a half-written message.
QueueHandle_t     xyQueue;

// oledMutex: a lock for the OLED screen.
// Any task that wants to draw on the screen must "take" the mutex first
// and "give" it back when done — this stops two tasks from drawing at once
// (which would corrupt the picture).
SemaphoreHandle_t oledMutex;


// ═══════════════════════════════════════════════════════════════════════════════
//  PIXEL ERROR MESSAGE
//  This tiny struct is what flows through xyQueue from taskSerial to taskServo.
//  It carries the nose position error in pixels from the frame centre.
// ═══════════════════════════════════════════════════════════════════════════════
struct XYMsg {
    int16_t err_x;   // positive = nose is to the RIGHT of centre → pan right
    int16_t err_y;   // positive = nose is BELOW centre → tilt down
};



// ═══════════════════════════════════════════════════════════════════════════════
//  GLOBAL RUNTIME STATE
//  Variables shared between tasks.  Marked `volatile` so the compiler doesn't
//  cache them in a register — any task reading the variable sees the latest value.
// ═══════════════════════════════════════════════════════════════════════════════

// Where the servos physically are right now (in ticks).
// Written by taskServo after each PID step; read by taskOLED and taskSerial.
volatile int     g_pan_tick    = PAN_ORIGIN;
volatile int     g_tilt_tick   = TILT_ORIGIN;

// Session control flags — changed by taskMenu, read by all other tasks.
volatile bool    g_running     = false;   // true = session is active (servos moving)
volatile bool    g_paused      = false;   // true = session paused (servos frozen)
volatile bool    g_calibMode   = false;   // true = currently running calibration
volatile bool    g_calibOK     = false;   // set to true by taskSerial when "CALIB_OK" arrives
volatile bool    g_targetLost  = false;   // true = Python said face/eyes are missing
volatile bool    g_autoStop    = false;   // true = Python said LOST or NOEYES → auto-stop

// Head pose message from Python.
//   0 = head is straight (OK)
//  +1 = turned right (yaw)   -1 = turned left (yaw)
//  +2 = tilted up (pitch)    -2 = tilted down (pitch)
volatile int     g_tiltMsg     = 0;

// Latest XYZ position in cm — computed by taskSerial, shown on OLED and sent to Python.
volatile float   g_xyz_x = 0.0f;
volatile float   g_xyz_y = 0.0f;
volatile float   g_xyz_z = 0.0f;

// Face distance (Z) from Python's DIST messages.
// g_dist_cm    = current distance — updated every 100 ms by taskSerial.
// g_calib_z_cm = the distance at the moment calibration was confirmed (our Z origin).
volatile float   g_dist_cm      = 50.0f;   // sensible default before first DIST arrives
volatile float   g_calib_z_cm   = 50.0f;


// ═══════════════════════════════════════════════════════════════════════════════
//  CALIBRATION DATA  (saved to NVS flash so it survives power-off)
//  These are the servo positions when the face was perfectly centred.
//  All XYZ coordinates are measured as deltas from these values.
// ═══════════════════════════════════════════════════════════════════════════════
int   calib_pan_tick   = PAN_ORIGIN;    // pan  tick at calibration
int   calib_tilt_tick  = TILT_ORIGIN;   // tilt tick at calibration


// ═══════════════════════════════════════════════════════════════════════════════
//  APP STATE MACHINE
//  The firmware is always in exactly one of these five states.
//  Transitions happen in taskMenu when the user turns or clicks the encoder.
// ═══════════════════════════════════════════════════════════════════════════════
enum AppState {
    ST_WELCOME,    // startup animation — waiting for it to finish
    ST_MENU,       // main menu — user chooses Calibrate or Start
    ST_CALIBRATE,  // calibration is running
    ST_RUNNING,    // session is active (tracking + logging)
    ST_STATS       // session ended — showing statistics screen
};
volatile AppState appState = ST_WELCOME;


// ═══════════════════════════════════════════════════════════════════════════════
//  MENU ITEMS
//  These are the text strings displayed in each menu.
//  The arrays are indexed by the current selection (menuSel, runMenuSel, etc.)
// ═══════════════════════════════════════════════════════════════════════════════

// Main menu (shown in ST_MENU): two choices
const char* menuItems[]      = { "Calibrate", "Start" };
#define MENU_COUNT 2
volatile int menuSel = 0;   // which item is currently highlighted

// In-session menu (shown in ST_RUNNING): pause/resume and stop
const char* runMenuItems[]   = { "Pause", "Stop" };
#define RUN_MENU_COUNT 2
volatile int runMenuSel = 0;

// Post-session menu (shown in ST_STATS): retry or recalibrate
const char* statsMenuItems[] = { "Retry", "Recalibrate" };
#define STATS_MENU_COUNT 2
volatile int statsMenuSel = 0;

// Short text shown on the OLED right panel ("Tracking", "Look at cam", etc.)
// Written by taskBuzzer, read by taskOLED.
char g_feedback[16] = "";

// Running session timer in whole seconds (updated by taskOLED while not paused).
volatile uint32_t g_sessionElapsedSec = 0;


// ═══════════════════════════════════════════════════════════════════════════════
//  SESSION STATISTICS
//  Accumulated during the session and displayed on the stats screen at the end.
// ═══════════════════════════════════════════════════════════════════════════════
uint32_t sessionStartMs = 0;   // millis() timestamp when the session started
struct {
    float    maxPan   = 0.0f;   // largest pan  angle reached (degrees from calib)
    float    maxTilt  = 0.0f;   // largest tilt angle reached (degrees from calib)
    uint32_t durSec   = 0;      // total session duration in seconds
} stats;


// ═══════════════════════════════════════════════════════════════════════════════
//  ENCODER STATE  (written by interrupt service routines, read by taskMenu)
//  The rotary encoder fires a hardware interrupt every time it clicks one step.
//  The ISR runs immediately — it's faster than any task.
// ═══════════════════════════════════════════════════════════════════════════════
volatile bool  encTurned = false;   // true = a rotation event is waiting to be processed
volatile int   encDir    = 0;       // +1 = clockwise, -1 = counter-clockwise
volatile bool  encClick  = false;   // true = the button was pressed

// encISR runs when ENC_CLK falls (one detent of rotation).
// We read ENC_DT at that exact moment to determine direction.
// IRAM_ATTR forces this code to live in fast RAM so it runs without any cache miss.
void IRAM_ATTR encISR() {
    encDir    = (digitalRead(ENC_DT) == HIGH) ? 1 : -1;
    encTurned = true;
}

// swISR runs when ENC_SW falls (button pressed).
// We only set the flag if it isn't already set — no need to count presses.
void IRAM_ATTR swISR() {
    if (!encClick) encClick = true;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  NVS LOAD / SAVE
//  NVS = Non-Volatile Storage — a tiny key/value store in the ESP32's flash.
//  Think of it like a permanent notepad that remembers things even after power-off.
//  We use it to save the calibration servo positions so you don't have to
//  recalibrate every time you turn on the device.
// ═══════════════════════════════════════════════════════════════════════════════

void loadNVS() {
    // Open the "PostureBOT" namespace in read-only mode.
    // getInt("cpan", PAN_ORIGIN) = read the value stored as "cpan";
    // if nothing is saved yet, use PAN_ORIGIN as the default.
    prefs.begin("PostureBOT", true);
    calib_pan_tick  = prefs.getInt("cpan",  PAN_ORIGIN);
    calib_tilt_tick = prefs.getInt("ctilt", TILT_ORIGIN);
    prefs.end();
}

void saveNVS() {
    // Open the "PostureBOT" namespace in read-write mode and store the current
    // calibration tick values under the keys "cpan" and "ctilt".
    prefs.begin("PostureBOT", false);
    prefs.putInt("cpan",  calib_pan_tick);
    prefs.putInt("ctilt", calib_tilt_tick);
    prefs.end();
}


// ═══════════════════════════════════════════════════════════════════════════════
//  OLED HELPERS
//  Small reusable functions for common drawing operations.
// ═══════════════════════════════════════════════════════════════════════════════

// oledTitle: draws a coloured banner across the top 16 pixels of the screen
// with the given text centred inside it.  Used at the top of every screen.
void oledTitle(const char* text, uint16_t bg, uint16_t fg) {
    tft.fillRect(0, 0, SCREEN_W, 16, bg);   // filled background rectangle
    tft.setTextColor(fg);
    tft.setTextSize(1);
    int tw = strlen(text) * 6;              // each character is 6 pixels wide at size 1
    tft.setCursor((SCREEN_W - tw) / 2, 4); // centre the text horizontally
    tft.print(text);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  WELCOME ANIMATION
//  Plays once on startup.  Draws expanding rainbow circles, then types out
//  the title letter-by-letter, shows a loading bar, and flashes "READY".
//  This is pure eye-candy — it doesn't do anything functional.
// ═══════════════════════════════════════════════════════════════════════════════
void drawWelcome() {
    tft.fillScreen(BLACK);

    const char* title = "PostureBOT";
    for (int i = 0; title[i]; i++) {
        tft.setTextSize(2);
        tft.setTextColor(CYAN);
        tft.setCursor(10 + i * 14, 18);
        tft.print(title[i]);
        delay(60);
    }

    tft.setTextSize(1);
    tft.setTextColor(MAGENTA);
    tft.setCursor(8, 46); tft.print("FACE TRACKER  v4");
    tft.setTextColor(DARK_CYAN);
    tft.setCursor(16, 58); tft.print("ESP32-S3 FreeRTOS");
    delay(300);

    tft.drawRect(12, 80, 104, 12, GREY);
    for (int p = 0; p <= 100; p++) {
        int bw = p * 102 / 100;
        uint16_t bc = (p < 40) ? CYAN : (p < 80) ? YELLOW : GREEN;
        tft.fillRect(13, 81, bw, 10, bc);
        tft.fillRect(12, 96, 104, 10, BLACK);
        tft.setTextColor(WHITE); tft.setTextSize(1);
        tft.setCursor(52, 96); tft.print(p); tft.print("%");
        delay(18);
    }

    for (int b = 0; b < 4; b++) {
        tft.setTextSize(2); tft.setTextColor(GREEN);
        tft.setCursor(26, 110); tft.print("READY");
        delay(280);
        tft.fillRect(0, 106, SCREEN_W, 22, BLACK);
        delay(160);
    }
    delay(200);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN MENU DRAW
//  Draws the two-item main menu (Calibrate / Start) with a highlight bar
//  on the currently selected item.  Called every time the selection changes.
// ═══════════════════════════════════════════════════════════════════════════════
void drawMenu(int sel = -1) {
    if (sel < 0) sel = menuSel;   // use the global selection if none specified
    xSemaphoreTake(oledMutex, portMAX_DELAY);
    tft.fillScreen(BLACK);
    oledTitle("[ PostureBOT ]", DARK_CYAN, BLACK);

    // Draw each menu item as a rounded rectangle.
    // The selected item gets a solid cyan fill; others get a dim outline only.
    for (int i = 0; i < MENU_COUNT; i++) {
        int y = 30 + i * 36;   // vertical position of this item
        if (i == sel) {
            tft.fillRoundRect(6, y, SCREEN_W - 12, 26, 5, CYAN);
            tft.setTextColor(BLACK); tft.setTextSize(1);
            tft.setCursor(14, y + 9);
            tft.print("\x10 "); tft.print(menuItems[i]);   // \x10 = ► arrow character
        } else {
            tft.drawRoundRect(6, y, SCREEN_W - 12, 26, 5, DARK_GREY);
            tft.setTextColor(GREY); tft.setTextSize(1);
            tft.setCursor(14, y + 9);
            tft.print("  "); tft.print(menuItems[i]);
        }
    }

    // Small hint text at the very bottom of the screen.
    tft.setTextColor(DARK_GREY);
    tft.setCursor(6, 120); tft.print("TURN=nav  PUSH=ok");
    xSemaphoreGive(oledMutex);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  STATS SCREEN
//  Shown after a session ends.  Displays duration, max pan/tilt angles,
//  a stability score, and two action buttons (Retry / Recalibrate).
// ═══════════════════════════════════════════════════════════════════════════════

// _drawStats: internal helper — renders the stats screen with `sel` highlighted.
void _drawStats(int sel) {
    stats.durSec = (millis() - sessionStartMs) / 1000;   // calculate final duration

    xSemaphoreTake(oledMutex, portMAX_DELAY);
    tft.fillScreen(BLACK);
    oledTitle("  SESSION COMPLETE  ", YELLOW, BLACK);

    // Vertical divider splits the screen into left (menu buttons) and right (numbers).
    tft.drawFastVLine(58, 16, SCREEN_H - 16, DARK_GREY);

    // Left panel: Retry / Recalibrate buttons
    for (int i = 0; i < STATS_MENU_COUNT; i++) {
        int y = 22 + i * 24;
        if (i == sel) {
            tft.fillRoundRect(2, y, 54, 18, 3, CYAN);
            tft.setTextColor(BLACK); tft.setTextSize(1);
            tft.setCursor(5, y + 5); tft.print(statsMenuItems[i]);
        } else {
            tft.drawRoundRect(2, y, 54, 18, 3, DARK_GREY);
            tft.setTextColor(GREY); tft.setTextSize(1);
            tft.setCursor(5, y + 5); tft.print(statsMenuItems[i]);
        }
    }
    tft.setTextColor(DARK_GREY); tft.setTextSize(1);
    tft.setCursor(2, 112); tft.print("TURN");
    tft.setCursor(2, 122); tft.print("PUSH=ok");

    // Right panel — duration in "Xm Ys" format
    uint32_t m = stats.durSec / 60, s = stats.durSec % 60;
    tft.setTextColor(WHITE); tft.setTextSize(1);
    tft.setCursor(62, 20); tft.print(m); tft.print("m"); tft.print(s); tft.print("s");

    // Pan and tilt max-deviation bars.
    // The bar is 64 px wide representing 85 degrees of movement (the full useful range).
    struct StatRow { const char* lbl; float val; uint16_t col; };
    StatRow rows[] = {
        { "Pan",  stats.maxPan,  CYAN    },
        { "Tilt", stats.maxTilt, MAGENTA },
    };
    for (int i = 0; i < 2; i++) {
        int y = 36 + i * 26;
        tft.setTextColor(rows[i].col); tft.setTextSize(1);
        tft.setCursor(62, y); tft.print(rows[i].lbl); tft.print(":");
        tft.setTextColor(WHITE); tft.print(rows[i].val, 1); tft.print("d");
        int bw = (int)constrain(rows[i].val * 64.0f / 85.0f, 0, 64);
        tft.fillRect(62, y + 10, bw, 4, rows[i].col);       // filled portion
        tft.drawRect(62, y + 10, 64, 4, DARK_GREY);         // empty bar outline
    }

    // Stability score: 100% = no movement; each degree of deviation costs 1.5 points.
    // Green ≥ 80, Yellow ≥ 50, Red < 50.
    float deviation = stats.maxPan + stats.maxTilt;
    int score = (int)constrain(100.0f - deviation * 1.5f, 0.0f, 100.0f);
    uint16_t scoreCol = (score >= 80) ? GREEN : (score >= 50) ? YELLOW : RED;
    tft.setCursor(62, 102); tft.setTextColor(DARK_GREY); tft.print("Stab:");
    tft.setTextColor(scoreCol); tft.setTextSize(1);
    tft.print(score); tft.print("%");

    xSemaphoreGive(oledMutex);
}

// drawStatsScreen: displays the stats and spins in a loop waiting for the user
// to pick an action (Retry or Recalibrate) by turning and clicking the encoder.
// Returns the index of the chosen menu item (0 = Retry, 1 = Recalibrate).
int drawStatsScreen() {
    statsMenuSel = 0;
    _drawStats(statsMenuSel);
    encClick = false;   // clear any stale button press
    for (;;) {
        esp_task_wdt_reset();
        if (encTurned) {
            encTurned    = false;
            statsMenuSel = (statsMenuSel + encDir + STATS_MENU_COUNT) % STATS_MENU_COUNT;
            _drawStats(statsMenuSel);
        }
        if (encClick) {
            encClick = false;
            return statsMenuSel;   // user confirmed their choice
        }
        delay(25);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  CALIBRATION PROCEDURE
//  Step-by-step what happens:
//   1. Reset PID state and drive servos to the mechanical origin (straight ahead).
//   2. Tell Python "CALIBRATING" so it starts sending ERR,px,py pixel errors.
//   3. Show an instruction screen ("Look at camera & hold still").
//   4. Wait — the ESP32 PID steers the camera while Python watches for the nose
//      to stay inside the target box for 5 seconds.
//   5. Python sends "CALIB_OK" when stable → we lock the current servo positions
//      as the calibration origin and save to flash.
// ═══════════════════════════════════════════════════════════════════════════════
void runCalibration() {
    g_calibMode = true;
    g_calibOK   = false;

    // Physically move both servos to the centre position first, then wait
    // 600 ms for them to settle before enabling the PID loop.
    setPanTick(PAN_ORIGIN);   setTiltTick(TILT_ORIGIN);
    g_pan_tick  = PAN_ORIGIN; g_tilt_tick = TILT_ORIGIN;
    delay(600);

    // Turn on g_running so taskServo starts processing ERR messages from Python.
    g_running = true;

    // Tell Python to start tracking — it will now send ERR,px,py every 50 ms.
    Serial.println("CALIBRATING");

    // Show the instruction screen so the user knows what to do.
    xSemaphoreTake(oledMutex, portMAX_DELAY);
    tft.fillScreen(BLACK);
    oledTitle("CALIBRATING", ORANGE, BLACK);
    tft.setTextColor(YELLOW); tft.setTextSize(1);
    tft.setCursor(8, 24); tft.print("Look at camera");
    tft.setCursor(8, 36); tft.print("& hold still.");
    tft.setCursor(8, 52); tft.print("Camera will track");
    tft.setCursor(8, 64); tft.print("your nose...");
    tft.setTextColor(DARK_GREY);
    tft.setCursor(8, 80); tft.print("Auto-locks when");
    tft.setCursor(8, 92); tft.print("face is centred.");
    tft.drawRoundRect(8, 104, 112, 11, 3, ORANGE);
    tft.setTextColor(ORANGE);
    tft.setCursor(46, 107); tft.print("Cancel");
    xSemaphoreGive(oledMutex);
    delay(400);

    // Flush any bounce or stale press left over from the menu-select click.
    encClick  = false;
    encTurned = false;

    // Animated wait loop — stays here until Python sends "CALIB_OK" or user cancels.
    uint32_t dotFrame = 0;
    while (!g_calibOK) {
        esp_task_wdt_reset();

        // Cancel calibration if encoder button is pressed.
        if (encClick) {
            encClick    = false;
            encTurned   = false;
            g_running   = false;
            g_calibMode = false;
            g_calibOK   = false;
            return;
        }

        dotFrame++;

        // Refresh the bottom of the OLED with a dot animation and Cancel button.
        xSemaphoreTake(oledMutex, portMAX_DELAY);
        tft.fillRect(0, 104, SCREEN_W, 24, BLACK);
        tft.setTextColor(CYAN); tft.setTextSize(1);
        tft.setCursor(8, 108); tft.print("Tracking");
        // Print 0-3 dots cycling: "Tracking" → "Tracking." → "Tracking.." → "Tracking..."
        for (uint32_t d = 0; d < (dotFrame % 4); d++) tft.print(".");

        // Cancel button — press the encoder knob to exit calibration.
        tft.drawRoundRect(8, 117, 112, 10, 3, ORANGE);
        tft.setTextColor(ORANGE);
        tft.setCursor(46, 120); tft.print("Cancel");
        xSemaphoreGive(oledMutex);

        delay(300);   // update display 3× per second
    }

    // "CALIB_OK" arrived — Python confirmed the face was stable for 5 seconds.
    // Freeze the PID by stopping the session, then capture the current positions.
    g_running = false;
    delay(100);   // give taskServo one more cycle to finish

    calib_pan_tick  = g_pan_tick;
    calib_tilt_tick = g_tilt_tick;
    g_calib_z_cm    = g_dist_cm;   // lock the face Z at this moment as the depth origin
    saveNVS();                     // write to flash so it persists after power-off
    Serial.println("CALIB_DONE");  // tell Python we're done

    // Show the result screen with the saved tick values and angles.
    xSemaphoreTake(oledMutex, portMAX_DELAY);
    tft.fillScreen(BLACK);
    oledTitle("CALIBRATED!", GREEN, BLACK);
    tft.setTextColor(CYAN); tft.setTextSize(1);
    tft.setCursor(8, 26); tft.print("Pan:  "); tft.print(calib_pan_tick);  tft.print(" ticks");
    tft.setCursor(8, 40); tft.print("Tilt: "); tft.print(calib_tilt_tick); tft.print(" ticks");

    float pan_deg  = (calib_pan_tick  - PAN_ORIGIN)  * PAN_DEG_PER_TICK;
    float tilt_deg = (calib_tilt_tick - TILT_ORIGIN) * TILT_DEG_PER_TICK;
    tft.setTextColor(WHITE);
    tft.setCursor(8, 56);
    if (pan_deg  >= 0) tft.print("+"); tft.print(pan_deg,  1); tft.print("deg pan");
    tft.setCursor(8, 68);
    if (tilt_deg >= 0) tft.print("+"); tft.print(tilt_deg, 1); tft.print("deg tilt");
    tft.setTextColor(LIME);
    tft.setCursor(8, 96); tft.print("Saved to NVS \x02");   // \x02 = ☻ smiley
    xSemaphoreGive(oledMutex);
    delay(2500);   // show the result for 2.5 seconds

    // Clear the calibration flags so the system returns to normal operation.
    g_calibOK   = false;
    g_calibMode = false;
    encClick    = false;
    encTurned   = false;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  STOP SESSION
//  Called when the user clicks "Stop" or when an auto-stop condition occurs
//  (face lost or eyes closed too long).  Freezes everything, shows stats,
//  then returns to the main menu.
// ═══════════════════════════════════════════════════════════════════════════════
void stopSession() {
    // Clear all session-related flags so taskServo and taskBuzzer go idle.
    g_running    = false;
    g_paused     = false;
    g_targetLost = false;
    g_autoStop   = false;
    g_tiltMsg    = 0;
    strcpy(g_feedback, "");

    // Show the stats screen and wait for the user to choose Retry or Recalibrate.
    appState = ST_STATS;
    int choice = drawStatsScreen();

    // Zero out stats so the next session starts fresh.
    memset(&stats, 0, sizeof(stats));

    if (choice == 1) {
        // User chose "Recalibrate" — run the calibration procedure before returning to menu.
        appState = ST_CALIBRATE;
        runCalibration();
    }
    // Either way, return to the main menu.
    appState = ST_MENU;
    menuSel  = 0;
    drawMenu();
}


// ═══════════════════════════════════════════════════════════════════════════════
//  TASK: SERIAL  (Core 1, Priority 4)
//  This task runs on the fast core and does two things every 5 ms:
//   RECEIVE: read all characters that arrived from Python over USB and parse
//            the message when a newline is found.
//   SEND: notify Python of session state changes (START / PAUSE / RESUME / STOP)
//         and broadcast the XYZ position every 100 ms.
// ═══════════════════════════════════════════════════════════════════════════════
void taskSerial(void* pv) {
    String     buf       = "";   // accumulates characters until we see '\n'
    bool       sentStart = false;   // tracks whether we've sent "START" this session
    bool       wasPaused = false;   // tracks whether we've sent "PAUSE" this session

    for (;;) {
        esp_task_wdt_reset();   // pat the watchdog so it doesn't reset the chip

        // ── RECEIVE ──────────────────────────────────────────────────────
        // Read characters one at a time until we see a newline ('\n').
        // Then trim whitespace and process the complete message.
        while (Serial.available()) {
            char c = (char)Serial.read();
            if (c == '\n') {
                buf.trim();

                if (buf == "WARN") {
                    // Face or eyes just went missing — trigger sweep mode in taskServo.
                    g_targetLost = true;

                } else if (buf == "FOUND") {
                    // Face/eyes came back — resume normal PID tracking.
                    g_targetLost = false;

                } else if (buf == "LOST" || buf == "NOEYES") {
                    // Grace period expired — face/eyes gone too long.
                    // Set g_autoStop so taskMenu triggers stopSession() on the next tick.
                    g_targetLost = false;
                    g_autoStop   = true;

                } else if (buf == "CALIB_OK") {
                    // Python confirmed the nose stayed stable for 5 s.
                    // runCalibration() is waiting on this flag.
                    g_calibOK = true;

                } else if (buf == "TILTL") {
                    // Head turned left — OLED will say "Turn Right" to correct it.
                    g_tiltMsg = -1;

                } else if (buf == "TILTR") {
                    // Head turned right — OLED will say "Turn Left".
                    g_tiltMsg = 1;

                } else if (buf == "PITCHUP") {
                    // Head pitched up — OLED will say "Tilt Down".
                    g_tiltMsg = 2;

                } else if (buf == "PITCHDOWN") {
                    // Head pitched down — OLED will say "Tilt Up".
                    g_tiltMsg = -2;

                } else if (buf == "TILTOK") {
                    // Head is back in the acceptable range — clear the message.
                    g_tiltMsg = 0;

                } else if (buf.startsWith("DIST,")) {
                    // Python is telling us how far away the face is.
                    // We store it and use it in the XYZ calculation below.
                    g_dist_cm = buf.substring(5).toFloat();

                } else if (buf.startsWith("ERR,")) {
                    // The main tracking message: "ERR,px,py"
                    // Parse the two integers and push them into the queue
                    // so taskServo's PID loop can consume them.
                    String vals = buf.substring(4);
                    int ci = vals.indexOf(',');
                    if (ci > 0) {
                        int16_t ex = (int16_t)vals.substring(0, ci).toInt();
                        int16_t ey = (int16_t)vals.substring(ci + 1).toInt();
                        g_targetLost = false;
                        XYMsg msg = { ex, ey };
                        xQueueSend(xyQueue, &msg, 0);   // 0 = don't wait if queue is full
                    }
                }
                buf = "";   // reset buffer for the next message
            } else {
                buf += c;   // keep building the message
            }
        }

        // ── XYZ COMPUTE + SEND  (every 100 ms during active session) ─────
        // We compute where the head is in 3-D space using:
        //   dp  = how many radians the pan servo has rotated from calib
        //   dt_a = how many radians the tilt servo has rotated from calib
        //   d    = current face distance in cm (from Python's DIST messages)
        // Then basic 3-D trigonometry gives X, Y, Z in cm.
        static uint32_t lastXyzMs = 0;
        if (g_running && !g_calibMode && !g_paused &&
                millis() - lastXyzMs >= 100) {
            lastXyzMs = millis();

            // Convert tick differences to angles in radians.
            float dp   = (float)(g_pan_tick  - calib_pan_tick)
                         * PAN_DEG_PER_TICK * (float)PI / 180.0f;
            float dt_a = (float)(g_tilt_tick - calib_tilt_tick)
                         * TILT_DEG_PER_TICK * (float)PI / 180.0f;
            float d    = g_dist_cm;

            // 3-D position: imagine the camera is a torch pointing in a direction.
            // The face is at distance d along that beam.
            float xv   = - d * sinf(dp);                    // left/right
            float yv   = d * sinf(dt_a);                 // up/down
            float zv   = d * cosf(dp) * cosf(dt_a) - g_calib_z_cm;  // depth relative to calib

            // Store for OLED display and send to Python for session logging.
            g_xyz_x = xv;  g_xyz_y = yv;  g_xyz_z = zv;
            char xyzBuf[48];
            snprintf(xyzBuf, sizeof(xyzBuf), "XYZ,%.1f,%.1f,%.1f", xv, yv, zv);
            Serial.println(xyzBuf);
        }

        // ── SEND session lifecycle messages ──────────────────────────────
        // We use local flags (sentStart, wasPaused) to send each message
        // exactly once when the condition becomes true — not every loop.
        if (g_running && !g_calibMode) {

            if (!sentStart) {
                // Session just started — tell Python to begin logging.
                Serial.println("START");
                sentStart = true;
                wasPaused = false;
            }

            if (g_paused && !wasPaused) {
                // Just paused — tell Python to stop adding data points.
                Serial.println("PAUSE");
                wasPaused = true;
            } else if (!g_paused && wasPaused) {
                // Just resumed — tell Python to continue logging.
                Serial.println("RESUME");
                wasPaused = false;
            }

        } else {
            if (sentStart) {
                // Session just ended — tell Python to finalise and save the session.
                Serial.println("STOP");
                sentStart = false;
                wasPaused = false;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(5));   // yield for 5 ms then repeat
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  TASK: SERVO  (Core 1, Priority 4)
//  Proportional-only control.
//
//  When tracking: drains xyQueue for the latest pixel error, computes target
//  tick positions with P control, then sweeps both servo axes step-by-step
//  (1 ms per step, both axes interpolated simultaneously) for fluid motion.
//
//  When target is lost: slow left/right pan sweep to hunt for the face.
// ═══════════════════════════════════════════════════════════════════════════════
void taskServo(void* pv) {
    XYMsg msg, latest;
    bool  got;
    int   sweepDir  = 1;
    int   sweepTick = PAN_ORIGIN;

    int   cur_pan  = PAN_ORIGIN;
    int   cur_tilt = TILT_ORIGIN;

    float    filt_err_x  = 0.0f;
    float    filt_err_y  = 0.0f;
    uint32_t warnHoldMs  = 0;      // timestamp when g_targetLost first became true

    for (;;) {
        esp_task_wdt_reset();

        if (!g_running || g_paused) {
            filt_err_x = 0.0f;  filt_err_y = 0.0f;
            warnHoldMs = 0;
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (g_targetLost) {
            // ── SWEEP MODE ─────────────────────────────────────────────
            // Face is missing.  Hold position for 500 ms first — minor movements
            // will self-correct without the camera sweeping away from the face.
            // Only start the left/right scan after the holdoff expires.
            filt_err_x = 0.0f;  filt_err_y = 0.0f;

            if (warnHoldMs == 0) warnHoldMs = millis();

            if (millis() - warnHoldMs >= 500) {
                sweepTick += sweepDir * 2;
                if (sweepTick >= PAN_MAX || sweepTick <= PAN_MIN)
                    sweepDir = -sweepDir;

                cur_pan  = constrain(cur_pan  + constrain(sweepTick - cur_pan,        -5, 5), PAN_MIN,  PAN_MAX);
                cur_tilt = constrain(cur_tilt + constrain(calib_tilt_tick - cur_tilt, -5, 5), TILT_MIN, TILT_MAX);
            }
            // else: hold current position and wait

        } else {
            warnHoldMs = 0;   // reset holdoff timer whenever face is present
            // ── P TRACKING MODE ────────────────────────────────────────
            // Drain the queue — only act on the most recent error.
            got = false;
            while (xQueueReceive(xyQueue, &msg, 0) == pdTRUE) {
                latest = msg; got = true;
            }

            if (got) {
                // Low-pass filter to smooth out MediaPipe landmark jitter.
                filt_err_x = 0.8f * filt_err_x + 0.2f * (float)latest.err_x;
                filt_err_y = 0.8f * filt_err_y + 0.2f * (float)latest.err_y;
                int16_t ex = (int16_t)filt_err_x;   // positive = nose right of centre
                int16_t ey = (int16_t)filt_err_y;   // positive = nose below centre

                // Apply deadzone per axis; negate ey for tilt direction.
                int pan_delta  = (abs(ex) > PID_DEADZONE_PX) ? (int)(PAN_KP  * (float)ex)   : 0;
                int tilt_delta = (abs(ey) > PID_DEADZONE_PX) ? (int)(TILT_KP * -(float)ey)  : 0;

                int target_pan  = constrain(cur_pan  + pan_delta,  PAN_MIN,  PAN_MAX);
                int target_tilt = constrain(cur_tilt + tilt_delta, TILT_MIN, TILT_MAX);

                int pan_steps  = target_pan  - cur_pan;
                int tilt_steps = target_tilt - cur_tilt;
                int total      = min(max(abs(pan_steps), abs(tilt_steps)), MAX_STEP_TICKS);

                // Sweep both axes simultaneously to their targets, 1 ms per step.
                // Linear interpolation keeps both axes arriving together.
                // MAX_STEP_TICKS caps how far we travel per update to prevent overshoot.
                for (int i = 1; i <= total; i++) {
                    esp_task_wdt_reset();
                    int p = cur_pan  + pan_steps  * i / total;
                    int t = cur_tilt + tilt_steps * i / total;
                    setPanTick(p);
                    setTiltTick(t);
                    g_pan_tick  = p;
                    g_tilt_tick = t;
                    delay(1);
                }

                cur_pan  = target_pan;
                cur_tilt = target_tilt;
            }
        }

        // Sync globals and physically write final position (handles sweep + idle).
        g_pan_tick  = cur_pan;
        g_tilt_tick = cur_tilt;
        setPanTick(cur_pan);
        setTiltTick(cur_tilt);

        if (g_running && !g_paused && !g_calibMode) {
            float panDeg  = fabsf((float)(cur_pan  - calib_pan_tick)  * PAN_DEG_PER_TICK);
            float tiltDeg = fabsf((float)(cur_tilt - calib_tilt_tick) * TILT_DEG_PER_TICK);
            if (panDeg  > stats.maxPan)  stats.maxPan  = panDeg;
            if (tiltDeg > stats.maxTilt) stats.maxTilt = tiltDeg;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  TASK: BUZZER  (Core 0, Priority 2)
//  Provides audio feedback by playing short tones.
//   - Double beep: face or eyes missing (target lost)
//   - Single chirp: head pose out of range (yaw/pitch warning)
//   - Silence: tracking normally
// ═══════════════════════════════════════════════════════════════════════════════
void taskBuzzer(void* pv) {
    // Initialise the LEDC (LED Control) peripheral in tone mode.
    // We reuse LEDC to drive the buzzer — it can generate any PWM frequency.
    ledcAttach(BUZZER_PIN, BUZZ_FREQ_HZ, BUZZ_RESOLUTION);

    for (;;) {
        esp_task_wdt_reset();

        // If the session is stopped, paused, or in calibration — stay silent.
        if (!g_running || g_paused || g_calibMode) {
            ledcWrite(BUZZER_PIN, 0);   // 0 = buzzer off
            if (g_running && g_paused) strcpy(g_feedback, "Paused");
            else if (!g_running)       strcpy(g_feedback, "");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        if (g_targetLost) {
            // ── Double beep: face or eyes missing ──
            // Two short beeps at 1800 Hz, then a 500 ms pause before repeating.
            strcpy(g_feedback, "Look at cam");
            ledcChangeFrequency(BUZZER_PIN, 1800, BUZZ_RESOLUTION);
            ledcWrite(BUZZER_PIN, 120); vTaskDelay(pdMS_TO_TICKS(100));
            ledcWrite(BUZZER_PIN,   0); vTaskDelay(pdMS_TO_TICKS(60));
            ledcWrite(BUZZER_PIN, 120); vTaskDelay(pdMS_TO_TICKS(100));
            ledcWrite(BUZZER_PIN,   0); vTaskDelay(pdMS_TO_TICKS(500));

        } else if (g_tiltMsg != 0) {
            // ── Single chirp: head pose warning ──
            // One short beep at 1400 Hz.  The feedback string describes the correction needed.
            const char* msg = "";
            if      (g_tiltMsg ==  1) msg = "Turn right";
            else if (g_tiltMsg == -1) msg = "Turn left";
            else if (g_tiltMsg ==  2) msg = "Tilt down";
            else if (g_tiltMsg == -2) msg = "Tilt up";
            strcpy(g_feedback, msg);
            ledcChangeFrequency(BUZZER_PIN, 1400, BUZZ_RESOLUTION);
            ledcWrite(BUZZER_PIN, 80); vTaskDelay(pdMS_TO_TICKS(60));
            ledcWrite(BUZZER_PIN,  0); vTaskDelay(pdMS_TO_TICKS(440));

        } else {
            // ── Normal tracking — no sound ──
            strcpy(g_feedback, "Tracking");
            ledcWrite(BUZZER_PIN, 0);
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  TASK: OLED  (Core 0, Priority 2)
//  Refreshes the display every 100 ms while a session is running (ST_RUNNING).
//  Other states (menu, calibration, stats) draw themselves when needed and
//  this task just sleeps.
//
//  LAYOUT (128×128 pixels):
//  ┌────────────────────────────────┐
//  │      title bar (16px)          │  ← status/warning message
//  ├──────────────┬─────────────────┤
//  │  LEFT (58px) │  RIGHT (68px)   │
//  │  Pause/Resume│  X: +00.0 cm    │
//  │  Stop        │  Y: +00.0 cm    │
//  │              │  Z: +00.0 cm    │
//  │  00m 00s     │  ─────────────  │
//  │              │  feedback msg   │
//  └──────────────┴─────────────────┘
// ═══════════════════════════════════════════════════════════════════════════════
void taskOLED(void* pv) {
    for (;;) {
        esp_task_wdt_reset();

        // Only draw the running screen when we're actually in a session.
        // All other states manage their own OLED content directly.
        if (appState != ST_RUNNING) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // Snapshot XYZ so we use consistent values for this entire frame.
        float xc = g_xyz_x, yc = g_xyz_y, zc = g_xyz_z;

        // Count up elapsed time only while not paused.
        if (!g_paused) {
            g_sessionElapsedSec = (millis() - sessionStartMs) / 1000;
        }
        uint32_t em = g_sessionElapsedSec / 60;   // minutes
        uint32_t es = g_sessionElapsedSec % 60;   // seconds

        xSemaphoreTake(oledMutex, portMAX_DELAY);
        tft.fillScreen(BLACK);

        // ── Title bar — changes colour and text based on current state ──
        if (g_paused) {
            oledTitle("  PAUSED  ", GREY, WHITE);
        } else if (g_targetLost) {
            oledTitle("FIND CAMERA", RED, WHITE);    // red = urgent: look at camera!
        } else if (g_tiltMsg ==  1) {
            oledTitle("TURN RIGHT", ORANGE, BLACK);
        } else if (g_tiltMsg == -1) {
            oledTitle("TURN LEFT",  ORANGE, BLACK);
        } else if (g_tiltMsg ==  2) {
            oledTitle("TILT DOWN",  ORANGE, BLACK);
        } else if (g_tiltMsg == -2) {
            oledTitle("TILT UP",    ORANGE, BLACK);
        } else {
            oledTitle("  TRACKING  ", DARK_CYAN, BLACK);   // normal: all good
        }

        // Vertical divider between the left menu and right data panels.
        tft.drawFastVLine(58, 16, SCREEN_H - 16, DARK_GREY);

        // ── Left panel: in-session action menu ──
        // The first button label switches between "Pause" and "Resume".
        const char* item0Label = g_paused ? "Resume" : "Pause";
        for (int i = 0; i < RUN_MENU_COUNT; i++) {
            int y = 22 + i * 30;
            const char* lbl = (i == 0) ? item0Label : runMenuItems[i];
            if (i == runMenuSel) {
                // Highlighted button: yellow when paused, cyan when running.
                tft.fillRoundRect(2, y, 54, 22, 3, g_paused ? YELLOW : CYAN);
                tft.setTextColor(BLACK); tft.setTextSize(1);
                tft.setCursor(6, y + 7); tft.print("\x10 "); tft.print(lbl);
            } else {
                tft.drawRoundRect(2, y, 54, 22, 3, DARK_GREY);
                tft.setTextColor(GREY); tft.setTextSize(1);
                tft.setCursor(6, y + 7); tft.print("  "); tft.print(lbl);
            }
        }

        // Session timer in bottom-left of the left panel.
        tft.setTextColor(g_paused ? GREY : WHITE); tft.setTextSize(1);
        tft.setCursor(4, 100);
        if (em < 10) tft.print("0"); tft.print(em); tft.print("m");
        if (es < 10) tft.print("0"); tft.print(es); tft.print("s");

        // Encoder hint at the very bottom.
        tft.setTextColor(DARK_GREY); tft.setTextSize(1);
        tft.setCursor(4, 118); tft.print("TURN=nav");
        tft.setCursor(4, 127); tft.print("PUSH=sel");

        // ── Right panel: live XYZ position ──
        // X — lateral (left/right), displayed in cyan.
        tft.setTextColor(DARK_GREY); tft.setTextSize(1);
        tft.setCursor(62, 20); tft.print("X");
        tft.setTextColor(g_paused ? DARK_GREY : CYAN);
        tft.setCursor(72, 20);
        if (xc >= 0) tft.print("+"); tft.print(xc, 1);
        tft.setTextColor(DARK_GREY); tft.setCursor(112, 20); tft.print("cm");

        // Y — vertical (up/down), displayed in magenta.
        tft.setTextColor(DARK_GREY); tft.setTextSize(1);
        tft.setCursor(62, 36); tft.print("Y");
        tft.setTextColor(g_paused ? DARK_GREY : MAGENTA);
        tft.setCursor(72, 36);
        if (yc >= 0) tft.print("+"); tft.print(yc, 1);
        tft.setTextColor(DARK_GREY); tft.setCursor(112, 36); tft.print("cm");

        // Z — depth, colour-coded by how far from calibration distance:
        //   green = close to calib (< 5 cm off)
        //   yellow = moderate drift (5-15 cm off)
        //   red = far from calib (> 15 cm off)
        float zAbs    = fabsf(zc);
        uint16_t zCol = g_paused ? DARK_GREY
                      : (zAbs < 5.0f  ? GREEN
                      : (zAbs < 15.0f ? YELLOW : RED));
        tft.setTextColor(DARK_GREY); tft.setTextSize(1);
        tft.setCursor(62, 52); tft.print("Z");
        tft.setTextColor(zCol);
        tft.setCursor(72, 52);
        if (zc >= 0) tft.print("+"); tft.print(zc, 1);
        tft.setTextColor(DARK_GREY); tft.setCursor(112, 52); tft.print("cm");

        // Thin horizontal rule separating the numbers from the feedback message.
        tft.drawFastHLine(60, 68, 66, DARK_GREY);

        // ── Feedback message from taskBuzzer ──
        // The string can be up to 15 characters.  If it's longer than 10,
        // we split it at the last space so it wraps onto two lines.
        tft.setTextColor(ORANGE); tft.setTextSize(1);
        char fbuf[16];
        strncpy(fbuf, g_feedback, 15); fbuf[15] = '\0';
        int flen = strlen(fbuf);
        if (flen > 10) {
            int sp = -1;
            for (int k = 10; k >= 0; k--) {
                if (fbuf[k] == ' ') { sp = k; break; }
            }
            if (sp > 0) {
                fbuf[sp] = '\0';
                tft.setCursor(62, 74); tft.print(fbuf);           // first line
                tft.setCursor(62, 86); tft.print(fbuf + sp + 1);  // second line after space
            } else {
                tft.setCursor(62, 74); tft.print(fbuf);
            }
        } else {
            tft.setCursor(62, 78); tft.print(fbuf);
        }

        // ── Stability dot ──
        // A small coloured circle in the corner.
        // drift = straight-line distance from the calibration X/Y origin.
        // green = barely moved, yellow = some movement, red = far from centre.
        float drift = sqrtf(xc * xc + yc * yc);
        uint16_t dotCol = (drift < 5.0f) ? GREEN : (drift < 15.0f) ? YELLOW : RED;
        tft.fillCircle(120, 112, 5, dotCol);
        tft.setTextColor(DARK_GREY); tft.setTextSize(1);
        tft.setCursor(62, 108); tft.print("Stab");

        xSemaphoreGive(oledMutex);
        vTaskDelay(pdMS_TO_TICKS(100));   // refresh 10 times per second
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  TASK: MENU  (Core 0, Priority 1)
//  The lowest-priority task — it watches the rotary encoder and drives the
//  app state machine.  Runs every 25 ms (fast enough to feel responsive).
// ═══════════════════════════════════════════════════════════════════════════════
void taskMenu(void* pv) {
    uint32_t lastClickMs = 0;   // used to debounce the encoder button

    for (;;) {
        esp_task_wdt_reset();

        // If an auto-stop condition arrived (face lost, eyes closed) and
        // we're still in ST_RUNNING, trigger the stop sequence.
        if (g_autoStop && appState == ST_RUNNING)
            stopSession();

        // ── Handle encoder rotation ──
        if (encTurned) {
            encTurned = false;
            if (appState == ST_MENU) {
                // Scroll through the main menu items (wraps around).
                menuSel = (menuSel + encDir + MENU_COUNT) % MENU_COUNT;
                drawMenu();
            } else if (appState == ST_RUNNING) {
                // Scroll through the in-session menu (Pause / Stop).
                runMenuSel = (runMenuSel + encDir + RUN_MENU_COUNT) % RUN_MENU_COUNT;
                // taskOLED will pick up the new runMenuSel on its next refresh.
            }
        }

        // ── Handle encoder button click ──
        if (encClick) {
            encClick = false;
            uint32_t now = millis();
            // Simple debounce: ignore clicks within 250 ms of the previous one.
            bool validClick = (now - lastClickMs >= 250);
            if (validClick) {
                lastClickMs = now;

                if (appState == ST_MENU) {
                    if (menuSel == 0) {
                        // "Calibrate" selected — run the calibration procedure,
                        // then return to the main menu.
                        appState = ST_CALIBRATE;
                        runCalibration();
                        appState = ST_MENU;
                        drawMenu();
                    } else if (menuSel == 1) {
                        // "Start" selected — initialise session state and begin tracking.
                        sessionStartMs      = millis();
                        g_sessionElapsedSec = 0;
                        memset(&stats, 0, sizeof(stats));
                        runMenuSel   = 0;
                        g_running    = true;
                        g_paused     = false;
                        g_autoStop   = false;
                        g_targetLost = false;
                        g_tiltMsg    = 0;
                        g_calibMode  = false;
                        strcpy(g_feedback, "");
                        appState = ST_RUNNING;
                        // taskSerial will send "START" to Python on its next loop.
                    }
                } else if (appState == ST_RUNNING) {
                    if (runMenuSel == 0) {
                        // Toggle pause: if running → pause; if paused → resume.
                        g_paused = !g_paused;
                    } else if (runMenuSel == 1) {
                        // "Stop" selected — end the session and show stats.
                        stopSession();
                    }
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(25));
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SETUP
//  Runs once when the ESP32 powers on.  Initialises all hardware, creates
//  the FreeRTOS objects, plays the welcome animation, then launches all tasks.
// ═══════════════════════════════════════════════════════════════════════════════
void setup() {
    // Start USB serial at 115200 baud — this is how Python talks to us.
    Serial.begin(115200);

    // Configure physical GPIO pins.
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(ENC_CLK,    INPUT_PULLUP);   // PULLUP = HIGH when not pressed
    pinMode(ENC_DT,     INPUT_PULLUP);
    pinMode(ENC_SW,     INPUT_PULLUP);

    // Attach hardware interrupts so encoder events are caught immediately.
    // FALLING = trigger when the pin goes from HIGH to LOW (contact made).
    attachInterrupt(digitalPinToInterrupt(ENC_CLK), encISR, FALLING);
    attachInterrupt(digitalPinToInterrupt(ENC_SW),  swISR,  FALLING);

    // Start SPI bus and initialise the OLED screen.
    // SPI.begin(SCK, MISO, MOSI, SS) — we don't use MISO so it's set to -1.
    SPI.begin(OLED_CLK, -1, OLED_MOSI, OLED_CS);
    tft.begin();
    tft.setRotation(0);
    tft.fillScreen(BLACK);   // clear to black on startup

    // Start I²C bus and initialise the PCA9685 servo driver.
    // Wire.begin(SDA, SCL) — pins 8 and 7 on this board.
    // The PCA9685 oscillator must be calibrated to 27 MHz for accurate servo timing.
    // setPWMFreq(50) sets 50 Hz — the standard frequency for hobby servos.
    Wire.begin(8, 7);
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(50);

    // Load the last calibration positions from flash (or use defaults if never saved).
    loadNVS();

    // Move both servos to the origin and wait 500 ms for them to physically arrive.
    g_pan_tick  = PAN_ORIGIN;
    g_tilt_tick = TILT_ORIGIN;
    setPanTick(g_pan_tick);
    setTiltTick(g_tilt_tick);
    delay(500);

    // Create FreeRTOS synchronisation objects.
    // xyQueue holds up to 20 XYMsg structs — the serial task fills it, the servo task drains it.
    xyQueue   = xQueueCreate(20, sizeof(XYMsg));
    // oledMutex prevents two tasks from calling tft.print() at the same time.
    oledMutex = xSemaphoreCreateMutex();

    // Disable the watchdog timer temporarily so the welcome animation can
    // run for several seconds without triggering a reset.
    esp_task_wdt_deinit();
    drawWelcome();

    // Re-enable the watchdog with a 10-second timeout.
    // If any task stops calling esp_task_wdt_reset() for 10 seconds straight,
    // the chip resets automatically — protection against infinite loops or crashes.
    const esp_task_wdt_config_t wdt_config = {
        .timeout_ms     = 10000,
        .idle_core_mask = 0,
        .trigger_panic  = true,   // trigger a full crash dump instead of silently resetting
    };
    esp_task_wdt_init(&wdt_config);
    esp_task_wdt_add(NULL);   // register the current (setup) task with the watchdog

    // Show the main menu now that hardware is ready.
    appState = ST_MENU;
    drawMenu();

    // Launch all FreeRTOS tasks.
    // xTaskCreatePinnedToCore(function, name, stackSize, param, priority, handle, core)
    //
    // Core 1 — latency-sensitive tasks that must respond to serial data fast.
    xTaskCreatePinnedToCore(taskSerial, "SER", 4096, NULL, 4, NULL, 1);
    xTaskCreatePinnedToCore(taskServo,  "SRV", 4096, NULL, 4, NULL, 1);
    // Core 0 — background tasks that can afford small delays.
    xTaskCreatePinnedToCore(taskBuzzer, "BUZ", 2048, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(taskOLED,   "OLE", 4096, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(taskMenu,   "MNU", 4096, NULL, 1, NULL, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  LOOP
//  Arduino's loop() function normally runs forever, but with FreeRTOS all the
//  real work is done in tasks.  loop() is just another task (the "loopTask")
//  and we don't need it — so we just pet the watchdog and sleep for 1 second.
// ═══════════════════════════════════════════════════════════════════════════════
void loop() {
    esp_task_wdt_reset();
    vTaskDelay(pdMS_TO_TICKS(1000));
}
