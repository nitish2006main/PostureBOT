# PostureBOT — Pan-Tilt Head Tracking and Monitoring System

PostureBOT is a desk device that watches your posture and gently corrects it.
A camera sits on a small motorized head (it can pan and tilt), tracks
your face, and physically follows you so it always knows where your
head is. If you slouch, lean too close to the screen, or turn away for
too long, it warns you with a buzzer and a screen message. Every session
gets saved so you can look back and see how your posture trends over
time.

The project has two halves that talk to each other over a USB cable:

- **Python (runs on your computer)** — uses the webcam and a
  face-tracking library (MediaPipe) to find your face, handles
  calibration, saves session data, and draws graphs afterward.
- **Firmware (runs on the ESP32-S3 microcontroller)** — controls the
  motors that move the camera, runs the small screen and buzzer, and
  reads the button/dial used to start and stop sessions.

---

## How it works

1. The computer turns on the webcam and looks for your face every
   frame, tracking the tip of your nose.
2. It figures out how far off-center your nose is and sends that to the
   ESP32 about 20 times a second.
3. The ESP32 uses that info to turn the motors and keep the camera
   pointed at your face.
4. It also looks at a few more points on your face (eyes, mouth, chin)
   to estimate which way your head is turned or tilted.
5. Based on how big your face looks in the frame, it estimates how far
   away you are from the screen.
6. **Calibration**: you hold your face still in a box on screen for 5
   seconds. That spot becomes "home base" — the ESP32 remembers this
   even after a restart.
7. As you move around, the ESP32 keeps sending back your position (in
   centimeters) compared to that home base, and the computer logs it.
8. If your face disappears or your eyes close for too long (5 seconds),
   the session pauses itself. If you tilt your head too far or lean too
   close/far, it buzzes and shows a message telling you how to fix it.
9. When the session ends, the computer draws a 3D graph showing how your
   head moved during the session and saves everything to a file.

---

## What's in this repo

| File                     | What it does                                                     |
|--------------------------|--------------------------------------------------------------------|
| `posturebot_v4.ino`       | Code that runs on the ESP32 — controls motors, screen, buzzer, button |
| `face_tracker_v4.py`      | Main program on your computer — camera, face tracking, calibration, saving sessions |
| `session_analytics.py`    | Lets you look back at old sessions — replay one, see trends, or compare a few side by side |

---

## Hardware used

- **Microcontroller**: ESP32-S3 (has two processor cores, runs
  FreeRTOS so it can do multiple things at once)
- **Screen**: small 128x128 color OLED display
- **Motor controller**: PCA9685 board that drives the two servo motors
  (one for left/right, one for up/down)
- **Input**: a rotary knob with a click button to navigate a simple menu
- **Buzzer**: small speaker for warning beeps
- **Camera**: any USB webcam plugged into the computer

### Wiring (ESP32-S3 pins)

| Part            | Pin |
|-----------------|-----|
| Screen MOSI     | 10  |
| Screen CLK      | 11  |
| Screen CS       | 9   |
| Screen DC       | 15  |
| Screen RST      | 3   |
| Buzzer          | 6   |
| Knob CLK        | 14  |
| Knob DT         | 13  |
| Knob button     | 12  |
| Motor board SDA | 8   |
| Motor board SCL | 7   |

### Keeping the motors safe

Every command sent to the motors passes through a safety check first, so
the code can never accidentally push a motor past its limit and break
it.

| Motor      | Home position | Min | Max |
|------------|----------------|-----|-----|
| Left/Right | 380            | 160 | 600 |
| Up/Down    | 355            | 190 | 585 |

---

## How the ESP32 code is organized

The ESP32 runs 5 small jobs at the same time instead of one big loop:

| Core | Job    | How often  | What it does                          |
|------|--------|------------|------------------------------------------|
| 1    | Serial | constant   | Talks to the computer over USB          |
| 1    | Servo  | constant   | Moves the motors to follow your face    |
| 0    | Buzzer | as needed  | Plays warning beeps                     |
| 0    | Screen | every 0.1s | Updates what's shown on the OLED        |
| 0    | Menu   | every 25ms | Reads the knob/button, updates the menu |

There's also a safety timer (watchdog) that automatically restarts the
chip if something ever freezes.

---

## Messages sent back and forth

**Computer → ESP32**

| Message                 | What it means                                     |
|--------------------------|-----------------------------------------------------|
| `ERR,px,py`              | How far off-center your nose is, in pixels         |
| `DIST,z_cm`               | How far away your face is, in cm                   |
| `CALIB_OK`                | Calibration is locked in — save this as home base  |
| `WARN`                    | Face or eyes not detected, giving it a few seconds |
| `FOUND`                   | Face/eyes found again                              |
| `LOST`                    | Face missing too long → stop the session           |
| `NOEYES`                  | Eyes closed too long → stop the session            |
| `TILTL` / `TILTR`         | Turned too far left/right                          |
| `PITCHUP` / `PITCHDOWN`   | Tilted head too far up/down                        |
| `TILTOK`                  | Head position is back to normal                    |
| `BUZZ_ON` / `BUZZ_OFF`    | Turn the warning buzzer on/off                     |

**ESP32 → Computer**

| Message       | What it means                              |
|----------------|-----------------------------------------------|
| `CALIBRATING` | Calibration has started                        |
| `CALIB_DONE`  | Calibration saved                              |
| `XYZ,x,y,z`   | Your current position (cm) vs. home base       |
| `START`       | Session started                                |
| `PAUSE`       | Session paused                                 |
| `RESUME`      | Session resumed                                |
| `STOP`        | Session ended                                  |

---

## What gets saved

Each session is saved as a file like
`sessions/session_2026-08-01_14-30-00.json` and includes:

- Every position you were tracked at during the session (with
  timestamps)
- Summary stats — how long the session was, how far you moved, average
  distance from the screen, and why it ended
- Any pause/resume breaks you took
- The calibration distance, so graphs can be redrawn later

### Looking at a single session

After a session ends (or when you replay one later), you get a 3D graph
showing your head movement like a little globe — the calibration point
in the middle, and a colored trail showing where your head went during
the session.

### Comparing sessions (`session_analytics.py`)

Running this file gives you 3 options:

1. **Replay** — see the 3D graph for one specific session again.
2. **Trend** — see a score (0–100%) for every session over time, with
   arrows showing whether each session was better or worse than the last
   one.
3. **Overlay** — pick 2 or more sessions and see their movement paths
   plotted on the same chart, so you can directly compare them.

The score is based on how far you drifted left/right and up/down during
the session — the less you moved, the higher the score.

One early test session is automatically skipped in the analytics because
a bug in that build made its distance readings 10x too big.

---

## How to set it up

### Firmware (ESP32 side)

1. Install the Arduino IDE and add ESP32-S3 board support.
2. Install these libraries: `Adafruit SSD1351`, `Adafruit GFX`,
   `Adafruit PWMServoDriver` (the `Preferences` library comes built in).
3. Wire everything up using the pin table above.
4. Upload `posturebot_v4.ino` to the board.

### Python (computer side)

```bash
pip install opencv-python mediapipe pyserial matplotlib numpy
```

1. Set which USB port to use in `face_tracker_v4.py` (or just let it
   auto-detect the board).
2. Run it:
   ```bash
   python face_tracker_v4.py
   ```
   The face-tracking model downloads automatically the first time you
   run it.
3. On the device, use the knob to pick **Calibrate**, hold still for 5
   seconds, then pick **Start** to begin tracking.
4. Press **Q** on your keyboard to close the program, or use the
   **Stop** option on the device to end a session and see your graph.
5. To look back at old sessions:
   ```bash
   python session_analytics.py
   ```

---

## A few extra details

- The motors are controlled directly by the ESP32, not the computer —
  so tracking stays smooth even if the computer lags for a moment.
- The camera reads frames in the background nonstop, so the program
  always works with the newest picture instead of a delayed one.
- If the USB connection drops, the program automatically tries to
  reconnect a few times before giving up.
