"""
face_tracker_v4.py  —  PostureBOT 
===========================================================
Camera is physically mounted on the pan-tilt head.
MediaPipe detects the nose tip pixel error and sends it to the ESP32.
The ESP32 runs the P controller and drives the servos directly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS FILE DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Opens the webcam and runs MediaPipe FaceLandmarker each frame.
  2. Extracts nose-tip (landmark 1) pixel position.
  3. Sends "ERR,px,py" pixel error to the ESP32 every 50 ms.
     The ESP32 runs P and drives servos autonomously.
  4. Calibration: Python detects when nose stays inside CALIB_BOX_PX for
     CALIB_HOLD_SECONDS (5 s), then sends "CALIB_OK".
  5. Sends "DIST,z_cm" every 100 ms so the ESP32 can compute XYZ.
  6. Receives "XYZ,x,y,z" from ESP32 and logs it to the session.
  7. Shows a 3D head-path graph after the session ends.
  8. Sends pose/warning state messages on state changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERIAL PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Python → ESP32:
    "ERR,px,py"   Raw pixel error from frame centre (nose position)
    "DIST,z_cm"   Current face Z distance in cm (for ESP32 XYZ)
    "CALIB_OK"    Stable nose lock — ESP32 saves calib ticks
    "WARN"        Face/eyes missing, grace window active
    "FOUND"       Face/eyes re-acquired
    "LOST"        Face missing ≥ 5 s → ESP32 auto-stops
    "NOEYES"      Eyes closed ≥ 5 s → ESP32 auto-stops
    "TILTL/R"     Yaw warning
    "PITCHUP/DOWN" Pitch warning
    "TILTOK"      All pose in range
    "BUZZ_ON"     Start buzzer (pose or distance warning active)
    "BUZZ_OFF"    Stop buzzer

  ESP32 → Python:
    "CALIBRATING"  Start calibration tracking
    "CALIB_DONE"   Calibration saved (informational)
    "XYZ,x,y,z"   Live position in cm (logged to session)
    "START"        Session started
    "PAUSE"        Session paused
    "RESUME"       Session resumed
    "STOP"         Session ended

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pip install opencv-python mediapipe pyserial matplotlib
"""

import cv2
import numpy as np
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import json
import os
import math
import threading
from datetime import datetime


# ══════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ══════════════════════════════════════════════════════════

SERIAL_PORT      = "COM3"
BAUD_RATE        = 115200
SEND_INTERVAL    = 0.05
SERIAL_BOOT_WAIT = 3.0
SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sessions"
)
# 5 s grace period before declaring face truly lost — allows brief look-aways.
GRACE_PERIOD_S = 5.0
DEADBAND_PX    = 15

CAM_WIDTH  = 640
CAM_HEIGHT = 480

CAMERA_FOV_H = 50.0
CAMERA_FOV_V = math.degrees(
    2 * math.atan(math.tan(math.radians(CAMERA_FOV_H / 2)) * (9 / 16))
)

PAN_ORIGIN  = 380;  PAN_MIN  = 160;  PAN_MAX  = 600
TILT_ORIGIN = 355;  TILT_MIN = 190;  TILT_MAX = 585

REAL_FACE_WIDTH_CM = 13.0  

CALIB_BOX_PX       = 60    # half-width; full box is 120×120 px
CALIB_HOLD_SECONDS = 5.0

YAW_THRESHOLD   = 20.0     # degrees
PITCH_THRESHOLD = 15.0     # degrees

# Buzzer thresholds
# Pose: buzz activates when outside the warning threshold; turns off only once the user
# is BUZZ_POSE_MARGIN_DEG inside that threshold, preventing rapid on/off toggling.
BUZZ_POSE_MARGIN_DEG = 5.0   # degrees of hysteresis inside YAW/PITCH_THRESHOLD

# Distance: buzz activates when the user deviates more than DIST_BUZZ_MARGIN_CM from
# their calibration distance. It turns off only once they return within DIST_BUZZ_RETURN_CM,
# giving a comfortable re-centering zone so it doesn't chatter near the boundary.
DIST_BUZZ_MARGIN_CM  = 15.0  # cm from calib z to trigger
DIST_BUZZ_RETURN_CM  = 8.0   # cm from calib z at which buzz turns off

# EAR below 0.20 = eye closed (vertical eyelid gap too small relative to width).
EAR_THRESHOLD = 0.20

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import urllib.request   # downloads the model file on first run

# 6 landmarks for head pose: nose tip, chin, right/left eye corner, right/left mouth corner.
POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Average adult face 3-D positions in mm for the 6 landmarks above.
#using nosetip as origin
#[x, y, z]
POSE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0  ],
    [0.0,  -63.6,  -12.5 ],
    [-43.3,  32.7,  -26.0],
    [43.3,   32.7,  -26.0],
    [-28.9, -28.9,  -24.1],
    [28.9,  -28.9,  -24.1],
], dtype=np.float64)

#used for detecting eyes closing and opening
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ── TRACKING ERROR METRICS ──────────────────────────────────

ERROR_LOG_FILE = "tracking_error_log.json"

# Ignore initial servo movement/transient response
SETTLED_STATE_DELAY_S = 5.0

tracking_error_data = []
tracking_start_time = None

# ══════════════════════════════════════════════════════════
#  SESSION STORAGE
# ══════════════════════════════════════════════════════════
#checks to see if session folder is created, creates it if not there
def ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_session(session: dict):
    ensure_sessions_dir()
    ts_clean = session.get("timestamp",
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ts_clean = ts_clean.replace(" ", "_").replace(":", "-")
    fpath = os.path.join(SESSIONS_DIR, f"session_{ts_clean}.json")

    out = dict(session)
    out["data"] = [
        {**pt, "elapsed_s": round(pt["elapsed_ms"] / 1000.0, 3)}
        for pt in session.get("data", [])
    ]
    try:
        with open(fpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Session] Saved → {fpath}")
    except Exception as e:
        print(f"[Session] Save error: {e}")

#shows all the stats
def finalise_session(session: dict) -> dict:
    data     = session.get("data", [])
    duration = time.time() - session["_start_time"]

    xs = [abs(p["x"]) for p in data]
    ys = [abs(p["y"]) for p in data]
    zs = [p["z"]      for p in data]

    session["stats"] = {
        "duration":    round(duration, 1),
        "max_x":       round(max(xs), 2) if xs else 0,
        "max_y":       round(max(ys), 2) if ys else 0,
        "max_z":       round(max(zs), 2) if zs else 0,
        "min_z":       round(min(zs), 2) if zs else 0,
        "avg_z":       round(sum(zs) / len(zs), 2) if zs else 0,
        "stop_reason": session.get("_stop_reason", "Manual"),
    }

    # Close any pause interval left open because the session ended while paused.
    pauses = session.get("paused_intervals", [])
    if pauses and pauses[-1]["resume_ms"] is None:
        pauses[-1]["resume_ms"] = round(duration * 1000)

    session.pop("_start_time",  None)
    session.pop("_stop_reason", None)
    return session

def calculate_tracking_error(error_data):
    """
    Calculates steady-state nose tracking error.
    """

    if len(error_data) == 0:
        print("No tracking data collected.")
        return

    errors = np.array([
        math.sqrt(p["err_x"]**2 + p["err_y"]**2)
        for p in error_data
    ])

    stats = {
        "samples": len(errors),
        "mean_error_px": round(float(np.mean(errors)), 2),
        "rms_error_px": round(float(np.sqrt(np.mean(errors**2))), 2),
        "max_error_px": round(float(np.max(errors)), 2),
        "std_px": round(float(np.std(errors)), 2)
    }

    print("\n========== TRACKING ERROR ==========")
    print(f"Samples: {stats['samples']}")
    print(f"Mean Error: {stats['mean_error_px']} px")
    print(f"RMS Error: {stats['rms_error_px']} px")
    print(f"Maximum Error: {stats['max_error_px']} px")
    print(f"Standard Deviation: {stats['std_px']} px")
    print("====================================")

    with open(ERROR_LOG_FILE, "w") as f:
        json.dump(
            {
                "statistics": stats,
                "raw_data": error_data
            },
            f,
            indent=4
        )

#convert int time into readable string time
def _fmt_duration(sec) -> str:
    s = int(round(float(sec or 0)))
    m, r = divmod(s, 60)
    return f"{m}m {r:02d}s" if m else f"{s}s"

#3D parametric graph
def show_session_graph(session: dict):
    """
    Plots the head path on a globe using longitude (pan) and latitude (tilt).

    Coordinate system
    -----------------
    Origin (0,0,0) = servo / pan-tilt head.
    Wireframe sphere radius = mean camera-to-face distance over the session.
    Longitude = pan angle  = atan2(x, abs_z)   — positive = head right
    Latitude  = tilt angle = atan2(y, abs_z)   — positive = head up
    abs_z[i]  = calib_z + z[i]  (absolute depth from servo; clamped ≥ 1 cm)
    Path inside sphere  → head moved closer than average.
    Path outside sphere → head moved farther than average.

    Blocks until the window is closed.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # ── STEP 1 — parse session ────────────────────────────────────
    data = session.get("data", [])
    if len(data) < 2:
        print("[Graph] Not enough data.")
        return

    xs = np.array([p["x"] for p in data])
    ys = np.array([p["y"] for p in data])   
    zs = np.array([p["z"] for p in data])
    ts = np.array([p["elapsed_ms"] / 1000.0 for p in data])

    calib_z = float(session.get("calib_z_cm", 0.0))
    if calib_z <= 1.0:
        calib_z = 60.0   # sensible fallback if not recorded

    # Absolute depth from servo — clamp to prevent atan2 instability when
    # the user moves closer than their calibration distance.
    abs_z = np.array([max(1.0, calib_z + z) for z in zs])

    # ── STEP 2 — spherical coordinates ───────────────────────────
    r        = np.sqrt(xs**2 + ys**2 + abs_z**2)
    sphere_r = float(np.mean(r))   # wireframe sphere radius = average distance

    # ── STEP 3 — figure & axes ───────────────────────────────────
    fig = plt.figure(figsize=(11, 9))
    fig.patch.set_facecolor("#040810")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#08111e")

    # ── STEP 4 — wireframe sphere at average distance ─────────────
    # Plot convention: ax.plot(pan_component, depth_component, tilt_component)
    # i.e. matplotlib X=pan, matplotlib Y=depth, matplotlib Z=tilt.
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(-np.pi / 2, np.pi / 2, 13)

    # Latitude rings (horizontal circles)
    for v_val in v:
        xr = sphere_r * np.cos(v_val) * np.cos(u)
        yr = sphere_r * np.sin(v_val) * np.ones_like(u)
        zr = sphere_r * np.cos(v_val) * np.sin(u)
        ax.plot(xr, zr, yr, color="#0e2236", alpha=0.3, linewidth=0.5)

    # Longitude meridians (vertical half-circles)
    for u_val in u[::2]:
        xm = sphere_r * np.cos(v) * np.cos(u_val)
        ym = sphere_r * np.sin(v)
        zm = sphere_r * np.cos(v) * np.sin(u_val)
        ax.plot(xm, zm, ym, color="#0e2236", alpha=0.3, linewidth=0.5)

    # Highlighted equator (lat=0) — calibration horizontal reference
    eq_x = sphere_r * np.cos(u)
    eq_z = sphere_r * np.sin(u)
    ax.plot(eq_x, eq_z, np.zeros_like(u),
            color="#1a3550", alpha=0.5, linewidth=0.8)

    # Highlighted prime meridian (lon=0) — calibration vertical reference
    pm_y = sphere_r * np.sin(v)
    pm_z = sphere_r * np.cos(v)
    ax.plot(np.zeros_like(v), pm_z, pm_y,
            color="#1a3550", alpha=0.5, linewidth=0.8)

    # ── STEP 5 — closest-approach sphere (if head moved significantly closer)
    min_abs_z = float(np.min(abs_z))
    if min_abs_z < calib_z * 0.85:
        close_r = float(np.min(r))
        for v_val in v[::3]:
            xr = close_r * np.cos(v_val) * np.cos(u)
            yr = close_r * np.sin(v_val) * np.ones_like(u)
            zr = close_r * np.cos(v_val) * np.sin(u)
            ax.plot(xr, zr, yr, color="#ff4444", alpha=0.08, linewidth=0.4)
        for u_val in u[::4]:
            xm = close_r * np.cos(v) * np.cos(u_val)
            ym = close_r * np.sin(v)
            zm = close_r * np.cos(v) * np.sin(u_val)
            ax.plot(xm, zm, ym, color="#ff4444", alpha=0.08, linewidth=0.4)
        ax.text(close_r, 0, 0, "Closest\napproach",
                color="#ff4444", fontsize=6, alpha=0.6)

    # ── STEP 6 — head path ────────────────────────────────────────
    colours = plt.cm.plasma(np.linspace(0, 1, len(xs)))
    for i in range(len(xs) - 1):
        ax.plot([xs[i],    xs[i + 1]],
                [abs_z[i], abs_z[i + 1]],
                [ys[i],    ys[i + 1]],
                color=colours[i], linewidth=2.0, alpha=0.9)

    # ── STEP 7 — reference markers ────────────────────────────────
    # Calibration origin: directly ahead on sphere surface
    ax.scatter(0, calib_z, 0, color="white", s=150, marker="*",
               zorder=5, label="Calibration origin")
    ax.scatter(xs[0], abs_z[0], ys[0], color="lime", s=100,
               zorder=5, label=f"t = 0s  (start)")
    ax.scatter(xs[-1], abs_z[-1], ys[-1], color="red", s=100,
               zorder=5, label=f"t = {ts[-1]:.1f}s  (end)")
    # Servo is at the true origin — the centre of the sphere
    ax.scatter(0, 0, 0, color="#00c896", s=200, marker="^",
               zorder=5, label="Servo (pan-tilt head)")
    # Dashed lines show direction vectors from the servo
    ax.plot([0, 0],      [0, calib_z],   [0, 0],
            color="#00c896", linewidth=0.8, alpha=0.4, linestyle="--")
    ax.plot([0, xs[0]],  [0, abs_z[0]],  [0, ys[0]],
            color="#00c896", linewidth=0.5, alpha=0.25, linestyle="--")
    ax.plot([0, xs[-1]], [0, abs_z[-1]], [0, ys[-1]],
            color="#00c896", linewidth=0.5, alpha=0.25, linestyle="--")

    # ── STEP 8 — lat/lon grid labels ─────────────────────────────
    # Longitude labels along equator (lat = 0)
    for lon_deg in [-60, -30, 0, 30, 60]:
        lon_rad = np.radians(lon_deg)
        lx = sphere_r * np.sin(lon_rad)
        lz = sphere_r * np.cos(lon_rad)
        label = f"{lon_deg}°" if lon_deg != 0 else "0° (forward)"
        ax.text(lx, lz, 0, label, color="#2a4a6a", fontsize=6, alpha=0.8)

    # Latitude labels along prime meridian (lon = 0)
    for lat_deg in [-45, -30, 0, 30, 45]:
        lat_rad = np.radians(lat_deg)
        ly = sphere_r * np.sin(lat_rad)
        lz = sphere_r * np.cos(lat_rad)
        label = f"{lat_deg}°" if lat_deg != 0 else "0° (level)"
        ax.text(0, lz, ly, label, color="#2a4a6a", fontsize=6, alpha=0.8)

    # ── Presentation ──────────────────────────────────────────────
    ax.set_axis_off()
    lim = sphere_r * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    sm = plt.cm.ScalarMappable(cmap="plasma",
         norm=plt.Normalize(vmin=ts[0], vmax=ts[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.45, pad=0.05)
    cbar.set_label("Time (s)", color="#b8d8e8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#2a4a6a")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#2a4a6a", fontsize=7)

    dur  = session.get("stats", {}).get("duration", ts[-1])
    pts  = len(data)
    stop = session.get("stats", {}).get("stop_reason", "Manual")
    ts_label = session.get("timestamp", "")
    fig.suptitle(
        f"Head Path  —  {ts_label}   |   {dur:.1f}s   |   {pts} pts   "
        f"|   Stop: {stop}   |   calib {calib_z:.0f} cm",
        color="#b8d8e8", fontsize=9, fontfamily="monospace"
    )
    ax.legend(loc="upper left", fontsize=8,
              facecolor="#08111e", edgecolor="#0e2236",
              labelcolor="white")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════
#  HEAD POSE + EAR HELPERS
# ══════════════════════════════════════════════════════════

def get_head_pose(landmarks, w: int, h: int, focal_px: int):
    """
    Returns (yaw, pitch, roll) in degrees using solvePnP against a 6-point
    3-D face model. Returns (None, None, None) if solving fails.
    """
    #convert the 6 landmarks from normalised coords to actual pixel positions
    img_pts = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in POSE_LANDMARK_IDS
    ], dtype=np.float64)

    #Camera matrix turning the points from the camera into pixel positions
    cam_mat = np.array([
        [focal_px, 0,     w / 2.0],
        [0,     focal_px, h / 2.0],
        [0,     0,     1.0    ]
    ], dtype=np.float64)
    #assume no lens distortion (not fully calibrated/finalized yet)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    #find the rotation that maps the 3D face model onto the 2D pixel points
    ok, rvec, _ = cv2.solvePnP(
        POSE_3D_MODEL, img_pts, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None, None

    #expand the compact rotation vector into a 3x3 rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    #extract yaw, pitch, roll in degrees from the rotation matrix
    pitch = math.degrees(math.asin(-rmat[2][1]))
    yaw   = math.degrees(math.atan2(rmat[2][0], rmat[2][2]))
    roll  = math.degrees(math.atan2(rmat[0][1], rmat[1][1]))
    return yaw, pitch, roll

#function for detecting the eye opening and closing
def eye_aspect_ratio(landmarks, eye_idx: list, w: int, h: int) -> float:
    """
    EAR = (v1 + v2) / (2 * h1): two vertical eyelid distances divided by
    horizontal eye width. Low value → eye closed.
    """
    #convert the 6 eye landmarks to pixel positions
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]

    #two vertical distances across the eyelid (left side and right side)
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    #horizontal width of the eye corner to corner
    h1 = math.dist(pts[0], pts[3])
    if h1 < 1e-6:
        return 1.0   # eye too small to measure — assume open
    #ratio drops toward 0 as the eye closes
    return (v1 + v2) / (2.0 * h1)


# ══════════════════════════════════════════════════════════
#  SERIAL HELPERS
# ══════════════════════════════════════════════════════════

def open_serial():
    """
    Opens SERIAL_PORT, or auto-detects any connected ESP32/Arduino if that fails.
    Returns the open Serial object, or None.
    """
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        time.sleep(SERIAL_BOOT_WAIT)
        print(f"[Serial] Connected → {SERIAL_PORT} @ {BAUD_RATE}")
        return ser
    except Exception as e:
        print(f"[Serial] Could not open {SERIAL_PORT}: {e}")

    print("[Serial] Trying auto-detect...")
    candidates = []
    for port in serial.tools.list_ports.comports():
        hwid = port.hwid.upper(); desc = port.description.upper()
        # 1A86 = CH340, 10C4 = CP2102, 2341 = Arduino
        if ("1A86" in hwid or "10C4" in hwid or "2341" in hwid
                or "CH34" in desc or "CP210" in desc
                or "UART" in desc or "USB-ENHANCED" in desc):
            candidates.append(port.device)

    for port in candidates:
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=0.01)
            time.sleep(SERIAL_BOOT_WAIT)
            print(f"[Serial] Auto-detected → {port}")
            return ser
        except Exception as e:
            print(f"[Serial] Auto-detect failed on {port}: {e}")

    print("[Serial] No ESP32 found. Available ports:")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device}  —  {p.description}  [{p.hwid}]")
    return None


def serial_send(ser, msg: str) -> bool:
    try:
        ser.write((msg + "\n").encode())
        return True
    except Exception as e:
        print(f"[Serial] Send error: {e}")
        return False


def serial_read_lines(ser) -> list:
    lines = []
    try:
        while ser.in_waiting:
            raw  = ser.readline()
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                lines.append(line)
    except Exception:
        pass
    return lines


def reconnect_serial(ser):
    print("[Serial] Connection lost — attempting reconnect...")
    try:
        ser.close()
    except Exception:
        pass
    for attempt in range(10):
        time.sleep(1.0)
        try:
            new_ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            time.sleep(1.0)
            print(f"[Serial] Reconnected on attempt {attempt + 1}")
            return new_ser
        except Exception as e:
            print(f"[Serial] Reconnect attempt {attempt + 1} failed: {e}")
    print("[Serial] Could not reconnect after 10 attempts.")
    return ser


# ══════════════════════════════════════════════════════════
#  THREADED CAMERA CAPTURE
# ══════════════════════════════════════════════════════════
class CameraCapture:
    """
    Reads frames from the camera continuously in a background thread so the
    buffer is always drained.  The main loop always gets the *latest* frame
    instead of a stale one that built up while MediaPipe was running.
    """
    def __init__(self, cap):
        self._cap     = cap
        self._frame   = None
        self._lock    = threading.Lock()
        self._stopped = False
        t = threading.Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)  # prevent tight spin when camera stalls

    def read(self):
        with self._lock:
            frame = self._frame       # grab reference only; copy outside lock
        if frame is None:
            return False, None
        return True, frame.copy()    # copy after releasing lock

    def stop(self):
        self._stopped = True


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    # collect all available serial port names on this machine for display purposes
    available = [p.device for p in serial.tools.list_ports.comports()]

    # print startup banner so the user knows the system is launching
    print("=" * 58)
    print("  PostureBOT")
    # show which serial port and baud rate will be used to talk to the ESP32
    print(f"  Port     : {SERIAL_PORT} @ {BAUD_RATE}")
    # show all ports detected so the user can troubleshoot if the wrong one is selected
    print(f"  Available: {available}")
    # show where session JSON files will be saved after each session ends
    print(f"  Sessions : {SESSIONS_DIR}")
    print("  Press Q in the camera window to quit")
    print("=" * 58)

    # make sure the sessions folder exists on disk before trying to save anything
    ensure_sessions_dir()

    # count how many previous sessions have already been recorded, shown on startup
    session_count = sum(
        1 for f in os.listdir(SESSIONS_DIR)
        if f.startswith("session_") and f.endswith(".json")
    )
    print(f"[Session] {session_count} previous session(s) in {SESSIONS_DIR}")

    # open the serial connection to the ESP32 — the program can't run without it
    ser = open_serial()
    if ser is None:
        # if no ESP32 is found, there's nothing to control so exit immediately
        print("[ERROR] Cannot open serial port."); return

    # path where the MediaPipe face landmark model will be stored locally
    MODEL_PATH = "face_landmarker.task"
    MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/latest/"
                  "face_landmarker.task")

    # download the model file from Google if it hasn't been downloaded yet
    if not os.path.exists(MODEL_PATH):
        print(f"[MediaPipe] Downloading model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[MediaPipe] Download complete.")

    # VIDEO mode uses motion between frames for smoother, more stable landmark tracking
    # compared to IMAGE mode which treats every frame independently
    base_opts = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    lm_opts   = mp_vision.FaceLandmarkerOptions(
        base_options                          = base_opts,
        running_mode                          = mp_vision.RunningMode.VIDEO,   # inter-frame smoothing
        num_faces                             = 1,                              # only track one person at a time
        min_face_detection_confidence         = 0.5,                           # minimum confidence to consider a face detected
        min_face_presence_confidence          = 0.5,                           # minimum confidence that a face is still present
        min_tracking_confidence               = 0.5,                           # minimum confidence to keep tracking vs re-detecting
        output_face_blendshapes               = False,                         # not needed — saves processing time
        output_facial_transformation_matrixes = False,                         # not needed — saves processing time
    )
    # create the actual face landmarker object that will process frames
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(lm_opts)

    # CAP_DSHOW is a Windows-specific low-latency camera driver that reduces frame delay
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # only keep 1 frame in the buffer so we always get the most recent image
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # MJPG compresses frames before sending over USB, keeping bandwidth low at high FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # set resolution to 640x480 — wide enough for accurate landmark detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    # request 30 frames per second from the camera
    cap.set(cv2.CAP_PROP_FPS, 30)

    # if the camera fails to open, close serial and exit cleanly
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); ser.close(); return

    # start the background camera thread so frames are always being read
    # even while MediaPipe or other processing is happening on the main thread
    camera = CameraCapture(cap)

    # create the OpenCV display window where the camera feed will be shown
    cv2.namedWindow("PostureBOT", cv2.WINDOW_NORMAL)
    # position the window at the top-left corner of the screen
    cv2.moveWindow("PostureBOT", 0, 0)

    # ── CALIBRATION STATE ─────────────────────────────────────────────────────
    calib_done          = False  # True once the 5s hold has completed
    calib_nose_in_since = None   # timestamp when nose first entered the calib box
    in_calibration      = False  # True while ESP32 is actively calibrating

    # focal px calculated from python script (calibrate_focal.py)
    FOCAL_PX    = 463  
    # estimated real-world distance from the camera to the user's face in centimetres
    current_z_cm = 0.0  # current estimated distance from camera to face in cm

    # ── SESSION STATE ─────────────────────────────────────────────────────────
    active_session    = None             # dict holding current session data, None if no session
    tracking_error_data = []            # tracking accuracy measurement
    tracking_start_time = None
    session_paused    = False            # True while session is paused
    pause_started_at  = None             # wall-clock time the current pause began
    total_paused_ms   = 0                # cumulative paused time this session in ms
    latest_xyz        = (0.0, 0.0, 0.0) # most recent XYZ position received from ESP32
    last_dist_t       = time.time()      # timestamp of last DIST message sent to ESP32

    # ── FACE AND EYE TRACKING STATE ───────────────────────────────────────────
    face_visible      = False  # True when MediaPipe currently sees a face
    face_lost_since   = None   # timestamp when face was last lost
    face_warn_sent    = False  # True if WARN has been sent for the current face loss
    eyes_closed       = False  # True when both eyes are currently below EAR threshold
    eyes_closed_since = None   # timestamp when eyes first closed
    eyes_warn_sent    = False  # True if WARN has been sent for the current eye closure
    in_warn           = False  # True while any grace-period warning is active
    last_pose_state   = "OK"   # last pose warning sent, prevents sending duplicates

    # depth at the moment calibration was confirmed — used as reference for distance alerts
    calib_z_cm    = 0.0    # z distance captured at calibration moment
    # ── BUZZER FLAGS ─────────────────────────────────────────────────────────
    buzz_pose     = False  # True while head pose is outside threshold
    buzz_dist     = False  # True while user is too far/close from calibration z
    buzz_eyes     = False  # True while both eyes are closed
    buzzer_on     = False  # tracks the current buzzer state sent to ESP32 to avoid redundant messages

    # ── TIMING AND FPS TRACKING ───────────────────────────────────────────────
    last_send_t       = time.time()  # timestamp of last serial message sent
    fps_last_update   = time.time()  # timestamp of last FPS calculation
    fps_frame_count   = 0            # frames counted since last FPS update
    fps_display       = 0.0          # displayed FPS value, updated every 0.5s

    # ── PER-FRAME WORKING STATE ───────────────────────────────────────────────
    # only one serial message is queued at a time — higher-priority messages overwrite lower ones
    pending     = None   # next serial message to send to ESP32
    lm_list     = []     # MediaPipe landmark results for the current frame
    frame_count = 0      # total frames processed, used to throttle EAR/pose checks
    avg_ear     = 1.0    # average eye aspect ratio across both eyes (1.0 = fully open)
    both_closed = False  # True when avg_ear drops below EAR_THRESHOLD
    yaw         = None   # current head yaw in degrees (None until first pose solve)
    pitch       = None   # current head pitch in degrees

    # ══════════════════════════════════════════════════════
    #  MAIN LOOP — runs once per camera frame
    # ══════════════════════════════════════════════════════
    while True:
        # grab the latest frame from the background camera thread
        ret, frame = camera.read()
        # if no frame is available yet (camera still starting up), try again
        if not ret:
            continue

        # the camera is physically mounted upside-down on the enclosure,
        # so rotate 180 degrees to correct the orientation
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # the camera is also mirrored, so flip horizontally to correct left/right
        frame = cv2.flip(frame, 1)

        # capture current time once and reuse throughout this frame to stay consistent
        now = time.time()

        # count this frame toward the FPS calculation
        fps_frame_count += 1
        # recalculate displayed FPS every 0.5 seconds to avoid jitter from single-frame timing
        if now - fps_last_update >= 0.5:
            fps_display = fps_frame_count / (now - fps_last_update)
            # reset counter and timer for the next 0.5s window
            fps_frame_count = 0; fps_last_update = now

        # get the pixel dimensions of the frame
        h, w   = frame.shape[:2]
        # compute the pixel coordinates of the frame centre — used as the tracking target
        cx, cy = w // 2, h // 2

        # ── PAUSED STATE ──────────────────────────────────────────────────────
        # when the session is paused, skip all tracking and only wait for RESUME or STOP
        if session_paused:
            # draw a black bar at the top of the frame for the status text
            cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
            # display paused status and instructions for resuming via the rotary encoder
            cv2.putText(frame,
                        "PostureBOT  |  PAUSED  |  TURN encoder to Resume/Stop",
                        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 200), 1)
            # display large PAUSED text in the centre of the frame
            cv2.putText(frame, "PAUSED",
                        (cx - 50, cy), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                        (80, 80, 200), 2)
            # show the paused frame to the user
            cv2.imshow("PostureBOT", frame)
            # allow quitting with Q even while paused
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # while paused, still read serial messages in case the user resumes or stops
            for line in serial_read_lines(ser):
                if line == "RESUME":
                    if active_session:
                        # calculate how many milliseconds have elapsed in the session up to this resume
                        ms = round(
                            (time.time() - active_session["_start_time"]) * 1000
                            - total_paused_ms)
                        ivs = active_session["paused_intervals"]
                        # close the open pause interval by recording the resume timestamp
                        if ivs and ivs[-1]["resume_ms"] is None:
                            ivs[-1]["resume_ms"] = ms
                        # add the duration of this pause to the running total of paused time
                        if pause_started_at is not None:
                            total_paused_ms += round(
                                (time.time() - pause_started_at) * 1000)
                        # clear the pause start time since we are no longer paused
                        pause_started_at = None
                    # mark the session as no longer paused so the main loop resumes
                    session_paused = False
                    print("[Session] Resumed")

                elif line == "STOP":
                    if active_session and active_session["data"]:
                        # if we stopped while paused, add the remaining pause time to the total
                        if pause_started_at is not None:
                            total_paused_ms += round(
                                (time.time() - pause_started_at) * 1000)
                        # compute final stats and write the session JSON file to disk
                        finished = finalise_session(active_session)
                        save_session(finished)
                        st = finished["stats"]
                        print(f"[Session] Saved  "
                              f"{len(finished['data'])} pts  "
                              f"{_fmt_duration(st['duration'])}  "
                              f"stop: {st['stop_reason']}")
                        # display the 3D head-path graph — blocks until the user closes the window
                        show_session_graph(finished)
                    # clear all session state so the system is ready for a new session
                    active_session   = None
                    latest_xyz       = (0.0, 0.0, 0.0)
                    session_paused   = False
                    pause_started_at = None
                    total_paused_ms  = 0
            # skip the rest of the loop while paused
            continue

        # ── FRAME PROCESSING ──────────────────────────────────────────────────
        # reset pixel error to zero at the start of each frame — will be filled in if a face is found
        err_x = err_y = 0.0
        # increment the frame counter used to throttle slow operations like EAR and pose
        frame_count += 1
        # convert the frame from BGR (OpenCV default) to RGB (MediaPipe requirement)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # use monotonic time for MediaPipe timestamps — guaranteed to never go backwards
        # which is required for VIDEO mode to work correctly
        frame_timestamp_ms = int(time.monotonic() * 1000)

        # wrap the RGB frame in MediaPipe's image container format
        lm_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # run face landmark detection on this frame — returns 478 landmark positions if a face is found
        lm_result   = face_landmarker.detect_for_video(lm_mp_image, frame_timestamp_ms)
        # extract the list of landmarks — empty list if no face was detected this frame
        lm_list     = lm_result.face_landmarks if lm_result.face_landmarks else []

        # ── FACE DETECTED ─────────────────────────────────────────────────────
        if lm_list:
            # use the first (and only) face's landmarks
            lm = lm_list[0]

            # landmark 1 is the nose tip — convert normalised 0-1 coords to actual pixel positions
            nose_x = int(lm[1].x * w)
            nose_y = int(lm[1].y * h)

            # landmarks 234 and 454 are the outer cheek edges — their pixel distance shrinks as the face moves away
            face_width_px = abs(lm[454].x - lm[234].x) * w
            if face_width_px > 1.0 and FOCAL_PX > 1.0:
                # pinhole camera model: real_size * focal_length / pixel_size = distance in cm
                current_z_cm = (REAL_FACE_WIDTH_CM * FOCAL_PX) / face_width_px

            # check if the user has moved too far or too close compared to their calibration position
            if calib_done and calib_z_cm > 0:
                # how many cm the user has drifted from their calibrated distance
                dist_dev = abs(current_z_cm - calib_z_cm)
                # activate distance buzzer if deviation exceeds the trigger threshold
                if dist_dev > DIST_BUZZ_MARGIN_CM and not buzz_dist:
                    buzz_dist = True
                # deactivate distance buzzer only once the user returns within the quieter return threshold
                # this prevents the buzzer from rapidly toggling on and off near the boundary
                elif dist_dev <= DIST_BUZZ_RETURN_CM and buzz_dist:
                    buzz_dist = False

            # pixel error = how far the nose is from the centre of the frame
            # positive err_x = nose is to the right of centre
            # positive err_y = nose is below centre (image Y increases downward)
            err_x = float(nose_x - cx)
            err_y = float(nose_y - cy)

            # ── SETTLED_STATE ERROR LOGGING ─────────────────────────────
            # Start timer once calibration is complete
            if calib_done and tracking_start_time is None:
                tracking_start_time = time.time()


            # Ignore first 5 seconds while P settles
            if tracking_start_time is not None:

                elapsed = time.time() - tracking_start_time

                if elapsed >= SETTLED_STATE_DELAY_S:

                    error_px = math.sqrt(
                        err_x**2 + err_y**2
                    )

                    tracking_error_data.append({
                        "time": round(elapsed, 3),
                        "err_x": round(err_x, 2),
                        "err_y": round(err_y, 2),
                        "error_px": round(error_px, 2)
                    })

            # if the face was previously lost and has just reappeared, reset warning state
            if not face_visible:
                face_visible    = True
                face_lost_since = None
                if face_warn_sent:
                    # notify the ESP32 that the face is back so it can cancel any pending stop
                    pending        = "FOUND"
                    face_warn_sent = False
                    in_warn        = False
                    # also reset eye warning since we're treating this as a fresh start
                    eyes_closed_since = None
                    eyes_warn_sent    = False

            # check if the nose is within the calibration target box at the centre of the frame
            nose_in_box = abs(err_x) <= CALIB_BOX_PX and abs(err_y) <= CALIB_BOX_PX

            # ── CALIBRATION HOLD LOGIC ────────────────────────────────────────
            if in_calibration:
                if calib_nose_in_since is None:
                    # nose just entered the box — start the hold timer
                    if nose_in_box:
                        calib_nose_in_since = now
                else:
                    # nose was already in the box — check if it drifted too far out
                    if (abs(err_x) > 2 * CALIB_BOX_PX or
                            abs(err_y) > 2 * CALIB_BOX_PX):
                        # nose drifted too far — reset the hold timer
                        calib_nose_in_since = None
                    elif now - calib_nose_in_since >= CALIB_HOLD_SECONDS:
                        # nose held steady for the full 5 seconds — calibration complete
                        # tell the ESP32 to save its current servo positions as the calibration origin
                        pending             = "CALIB_OK"
                        in_calibration      = False
                        calib_done          = True
                        # record the face depth at the moment of calibration as the reference distance
                        calib_z_cm          = current_z_cm
                        calib_nose_in_since = None
                        print(f"[Calib] Done — z={current_z_cm:.1f}cm")

            # ── EYE STATE AND HEAD POSE ───────────────────────────────────────
            # EAR and pose change slowly so recalculating every 10th frame is enough
            # this reduces CPU load significantly without affecting responsiveness
            if frame_count % 10 == 0:
                # compute Eye Aspect Ratio for both eyes separately then average them
                left_ear    = eye_aspect_ratio(lm, LEFT_EYE_IDX,  w, h)
                right_ear   = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
                avg_ear     = (left_ear + right_ear) / 2.0
                # if average EAR drops below threshold, both eyes are considered closed
                both_closed = avg_ear < EAR_THRESHOLD
                # mirror the eye state to the buzzer flag
                buzz_eyes   = both_closed

            # ── EYE CLOSURE WARNING ───────────────────────────────────────────
            if both_closed:
                if not eyes_closed:
                    # eyes just closed — record the timestamp so we can measure how long they stay closed
                    eyes_closed       = True
                    eyes_closed_since = now
                # how long the eyes have been continuously closed
                elapsed_eyes = now - eyes_closed_since
                if elapsed_eyes < GRACE_PERIOD_S and not eyes_warn_sent:
                    # eyes have been closed for less than 5s — send a warning to the ESP32
                    pending        = "WARN"
                    eyes_warn_sent = True
                    in_warn        = True
                elif elapsed_eyes >= GRACE_PERIOD_S and eyes_warn_sent:
                    # eyes have been closed for more than 5s — tell ESP32 to auto-stop the session
                    pending        = "NOEYES"
                    if active_session:
                        # record why the session stopped for the saved stats
                        active_session["_stop_reason"] = "Eyes closed"
                    # reset eye warning state ready for next time
                    eyes_closed_since = None
                    eyes_warn_sent    = False
                    in_warn           = False
            else:
                # eyes are open — clear the eye closed warning if one was active
                if eyes_closed:
                    eyes_closed       = False
                    eyes_closed_since = None
                    if eyes_warn_sent:
                        # notify ESP32 that eyes are open again
                        pending        = "FOUND"
                        eyes_warn_sent = False
                        in_warn        = False

            # ── HEAD POSE WARNING ─────────────────────────────────────────────
            # recalculate head pose every 10th frame — same throttle as EAR
            if frame_count % 10 == 0:
                yaw, pitch, _ = get_head_pose(lm, w, h, FOCAL_PX)
            if yaw is not None:
                # send a directional warning only when the state actually changes
                # to avoid flooding the ESP32 with repeated identical messages
                if   yaw >  YAW_THRESHOLD   and last_pose_state != "YAW_L":
                    # head turned too far left (from the user's perspective)
                    pending         = "TILTL";  last_pose_state = "YAW_L"
                elif yaw < -YAW_THRESHOLD   and last_pose_state != "YAW_R":
                    # head turned too far right
                    pending         = "TILTR";  last_pose_state = "YAW_R"
                elif pitch >  PITCH_THRESHOLD and last_pose_state != "PITCH_UP":
                    # head tilted too far up
                    pending         = "PITCHUP"; last_pose_state = "PITCH_UP"
                elif pitch < -PITCH_THRESHOLD and last_pose_state != "PITCH_DN":
                    # head tilted too far down
                    pending         = "PITCHDOWN"; last_pose_state = "PITCH_DN"
                elif (abs(yaw) <= YAW_THRESHOLD and
                      abs(pitch) <= PITCH_THRESHOLD and
                      last_pose_state != "OK"):
                    # head is back within acceptable range — send all-clear
                    pending         = "TILTOK"; last_pose_state = "OK"

                # pose buzzer uses hysteresis: activates when outside threshold,
                # but only deactivates once the user is BUZZ_POSE_MARGIN_DEG inside it
                # this prevents rapid buzzer toggling when the head hovers near the boundary
                pose_bad  = abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD
                pose_good = (abs(yaw)   <= YAW_THRESHOLD   - BUZZ_POSE_MARGIN_DEG and
                             abs(pitch) <= PITCH_THRESHOLD - BUZZ_POSE_MARGIN_DEG)
                if pose_bad and not buzz_pose:
                    buzz_pose = True
                elif pose_good and buzz_pose:
                    buzz_pose = False

            # ── SEND PIXEL ERROR ──────────────────────────────────────────────
            # if no high-priority message is waiting, queue the nose pixel error for the ESP32
            # the ESP32 uses this error to drive the P controller and move the servos
            if pending not in ("CALIB_OK", "FOUND", "TILTL", "TILTR",
                               "PITCHUP", "PITCHDOWN", "TILTOK"):
                # apply deadband: zero out any error smaller than DEADBAND_PX pixels
                # this prevents the servos from jittering when the nose is nearly centred
                dx = int(err_x) if abs(err_x) > DEADBAND_PX else 0
                dy = int(err_y) if abs(err_y) > DEADBAND_PX else 0
                pending = f"ERR,{dx},{dy}"

            # colour the tracking overlays orange/red if a warning is active, green if all clear
            box_col = (0, 80, 255) if in_warn else (0, 255, 128)

            # draw cyan dots on the 6 landmarks used for head pose estimation
            for idx in POSE_LANDMARK_IDS:
                cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)),
                           2, (0, 200, 255), -1)
            # draw green dots on the 12 landmarks used for eye state detection
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)),
                           2, (0, 160, 80), -1)

            # draw a crosshair directly on the nose tip to show what is being tracked
            cv2.drawMarker(frame, (nose_x, nose_y),
                           box_col, cv2.MARKER_CROSS, 14, 2)

            # build the pose string showing yaw and pitch angles — empty if pose not yet solved
            pose_str = ""
            if yaw is not None:
                pose_str = f"Y:{yaw:+.0f} P:{pitch:+.0f}"
            # display pixel error and head angles in the bottom-left corner of the frame
            cv2.putText(frame,
                        f"err({err_x:+.0f},{err_y:+.0f})  {pose_str}",
                        (8, h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_col, 1)

            # display EAR value and eye open/closed state — turns red when eyes are closed
            ear_col = (0, 80, 255) if both_closed else (0, 200, 80)
            cv2.putText(frame,
                        f"EAR:{avg_ear:.2f} {'CLOSED' if both_closed else 'open'}",
                        (8, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, ear_col, 1)

            # if eyes are closed and still within the grace period, show countdown on screen
            if in_warn and eyes_closed_since:
                rem = max(0.0, GRACE_PERIOD_S - (now - eyes_closed_since))
                cv2.putText(frame, f"OPEN EYES  {rem:.1f}s",
                            (cx - 80, cy + 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 80, 255), 2)

            # if the user has drifted too far from their calibrated distance, show which way to move
            if buzz_dist and calib_z_cm > 0:
                # positive dist_dev means user is farther away than calibration (moved back)
                # negative dist_dev means user is closer than calibration (leaned in)
                dist_dev = current_z_cm - calib_z_cm
                dist_msg = (f"MOVE CLOSER  {abs(dist_dev):.0f}cm" if dist_dev > 0
                            else f"MOVE BACK  {abs(dist_dev):.0f}cm")
                cv2.putText(frame, dist_msg,
                            (cx - 100, cy + 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 80, 255), 2)

        # ── NO FACE DETECTED ──────────────────────────────────────────────────
        else:
            # face was visible last frame but disappeared this frame
            if face_visible:
                face_visible        = False
                # record when the face was lost so we can time the grace period
                face_lost_since     = now
                # clear all eye and pose state since there is no face to measure
                eyes_closed         = False
                eyes_closed_since   = None
                eyes_warn_sent      = False
                calib_nose_in_since = None
                # deactivate all buzzers since we can no longer measure anything
                buzz_pose           = False
                buzz_dist           = False
                buzz_eyes           = False

            # face is still missing — manage the grace period and warning escalation
            if face_lost_since is not None:
                elapsed = now - face_lost_since
                if elapsed < GRACE_PERIOD_S and not face_warn_sent:
                    # face missing for less than 5s — send a soft warning, don't stop yet
                    pending        = "WARN"
                    face_warn_sent = True
                    in_warn        = True
                elif elapsed >= GRACE_PERIOD_S and face_warn_sent:
                    # face missing for more than 5s — tell ESP32 to auto-stop the session
                    pending        = "LOST"
                    if active_session:
                        # record why the session stopped
                        active_session["_stop_reason"] = "Face lost"
                    # reset face loss state ready for next detection
                    face_lost_since = None
                    face_warn_sent  = False
                    in_warn         = False

            # show a countdown while in the grace period, or TARGET LOST once time expires
            if in_warn and face_lost_since:
                # calculate remaining grace period time
                rem = max(0.0, GRACE_PERIOD_S - (now - face_lost_since))
                cv2.putText(frame, f"LOOK AT CAMERA  {rem:.1f}s",
                            (cx - 110, cy - 16),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 80, 255), 2)
                # animate a pulsing circle at the centre to draw the user's attention
                pr = int(22 + 10 * abs(math.sin(now * 3)))
                cv2.circle(frame, (cx, cy), pr, (0, 0, 200), 1)
            else:
                # grace period expired — face has been lost for too long
                cv2.putText(frame, "TARGET LOST",
                            (cx - 68, cy - 16),
                            cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 255), 1)

        # ── BUZZER SYNC ───────────────────────────────────────────────────────
        # combine all three buzzer conditions — any one of them activates the buzzer
        new_buzzer_on = buzz_pose or buzz_dist or buzz_eyes
        if new_buzzer_on != buzzer_on:
            # only send a message if the buzzer state actually changed — avoids serial spam
            buzzer_on = new_buzzer_on
            serial_send(ser, "BUZZ_ON" if buzzer_on else "BUZZ_OFF")

        # ── SERIAL SEND ───────────────────────────────────────────────────────
        # send the queued message to the ESP32, but no faster than once per SEND_INTERVAL (50ms)
        # this prevents flooding the ESP32's serial buffer
        if pending is not None and now - last_send_t >= SEND_INTERVAL:
            ok = serial_send(ser, pending)
            if not ok:
                # serial send failed — attempt to reconnect to the ESP32
                ser = reconnect_serial(ser)
            pending     = None
            last_send_t = now

        # send the current face depth to the ESP32 every 100ms after calibration is complete
        # the ESP32 uses this to compute real-world XYZ coordinates from its servo angles
        if calib_done and now - last_dist_t >= 0.1:
            last_dist_t = now
            serial_send(ser, f"DIST,{current_z_cm:.1f}")

        # ── SERIAL RECEIVE ────────────────────────────────────────────────────
        # read all messages waiting in the serial buffer from the ESP32
        for line in serial_read_lines(ser):

            # ESP32 has entered calibration mode — reset all calibration state on the Python side
            if line == "CALIBRATING":
                in_calibration      = True
                calib_done          = False
                calib_z_cm          = 0.0
                calib_nose_in_since = None
                # clear buzzer flags since calibration needs a clean start
                buzz_pose           = False
                buzz_dist           = False
                print("[Calib] Started — ESP32 P driving to nose, then 5 s hold")

            elif line == "CALIB_DONE":
                # ESP32 has saved its servo positions to flash memory as the calibration origin
                print("[Calib] ESP32 saved calibration ticks")

            # ESP32 started a new posture tracking session
            elif line == "START":
                # create a new session dictionary to hold all data for this session
                active_session = {
                    "timestamp":        datetime.now().strftime(
                                            "%Y-%m-%d %H:%M:%S"),  # human-readable start time
                    "calib_z_cm":       round(calib_z_cm, 1),      # reference distance at calibration
                    "data":             [],                          # list of XYZ data points
                    "stats":            {},                          # filled in when session ends
                    "paused_intervals": [],                          # list of pause/resume timestamps
                    "_start_time":      time.time(),                 # wall-clock start for duration tracking
                    "_stop_reason":     "Manual",                    # default stop reason, overwritten if auto-stopped
                }
                session_paused   = False
                pause_started_at = None
                total_paused_ms  = 0
                # reset depth send timer so depth messages start immediately
                last_dist_t      = time.time()
                print(f"[Session] Started  {active_session['timestamp']}")

            # ESP32 user pressed pause on the rotary encoder
            elif line == "PAUSE":
                if active_session:
                    # calculate elapsed active time (excluding previous pauses) in milliseconds
                    ms = round(
                        (time.time() - active_session["_start_time"]) * 1000
                        - total_paused_ms)
                    # record this pause interval with an open resume_ms (filled in on resume)
                    active_session["paused_intervals"].append(
                        {"pause_ms": ms, "resume_ms": None})
                    # record when this pause started so we can measure its duration
                    pause_started_at = time.time()
                    session_paused   = True
                    print(f"[Session] Paused at {ms} ms")

            # ESP32 user ended the session via the rotary encoder
            elif line == "STOP":
                if active_session and active_session["data"]:
                    # compute final statistics and write the session to a JSON file
                    finished = finalise_session(active_session)
                    save_session(finished)
                    st = finished["stats"]
                    print(f"[Session] Saved  "
                          f"{len(finished['data'])} pts  "
                          f"{_fmt_duration(st['duration'])}  "
                          f"stop: {st['stop_reason']}")
                    # show the 3D head-path globe — blocks until the user closes the matplotlib window
                    show_session_graph(finished)
                elif active_session:
                    # session was started but no XYZ data was ever logged — discard it
                    print("[Session] Stopped with no data — discarded.")
                # clear session state so the system is ready for a new session
                active_session = None
                latest_xyz     = (0.0, 0.0, 0.0)

            # ESP32 is broadcasting the current 3D head position in cm
            elif line.startswith("XYZ,"):
                parts = line.split(",")
                if len(parts) == 4:
                    try:
                        # parse the three coordinate values from the comma-separated message
                        x = round(float(parts[1]), 2)
                        y = round(float(parts[2]), 2)
                        z = round(float(parts[3]), 2)
                        # update the live display readout with the latest position
                        latest_xyz = (x, y, z)
                        # only log to the session if a session is active and not paused
                        if active_session and not session_paused:
                            # calculate active elapsed time in ms, excluding all pause durations
                            ms = round(
                                (now - active_session["_start_time"]) * 1000
                                - total_paused_ms)
                            # compute radial distance from origin for later stats
                            r = round(math.sqrt(x**2 + y**2 + z**2), 2)
                            # append this data point to the session log
                            active_session["data"].append({
                                "x": x, "y": y, "z": z, "r": r,
                                "elapsed_ms": ms
                            })
                    except ValueError:
                        # malformed XYZ message — skip it silently
                        pass

        # ── OVERLAYS ──────────────────────────────────────────────────────────
        # draw a small static crosshair at the exact centre of the frame
        # this shows the user where their nose needs to be to stay centred
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (60, 80, 80), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (60, 80, 80), 1)

        # ── CALIBRATION OVERLAY ───────────────────────────────────────────────
        if in_calibration:
            # position the calibration box around the nose tip, not the frame centre
            npx = cx + int(err_x)
            npy = cy + int(err_y)
            # check whether the nose is currently inside the calibration box
            nose_in_box    = face_visible and abs(err_x) <= CALIB_BOX_PX and abs(err_y) <= CALIB_BOX_PX
            # green box when nose is inside, blue when outside
            calib_rect_col = (0, 255, 80) if nose_in_box else (0, 180, 255)
            if face_visible:
                # draw the calibration target box around the nose tip
                cv2.rectangle(frame,
                              (npx - CALIB_BOX_PX, npy - CALIB_BOX_PX),
                              (npx + CALIB_BOX_PX, npy + CALIB_BOX_PX),
                              calib_rect_col, 2)
            if calib_nose_in_since is not None:
                # nose is inside the box — show hold progress bar at the bottom of the frame
                held   = min(now - calib_nose_in_since, CALIB_HOLD_SECONDS)
                remain = CALIB_HOLD_SECONDS - held
                # bar width is proportional to how long the nose has been held inside the box
                bar_px = int(w * held / CALIB_HOLD_SECONDS)
                # dark green background bar
                cv2.rectangle(frame, (0, h - 18), (w, h), (20, 50, 20), -1)
                # bright green progress fill
                cv2.rectangle(frame, (0, h - 18), (bar_px, h), (0, 255, 80), -1)
                # show countdown timer so the user knows how long to hold still
                cv2.putText(frame, f"HOLD STILL  {remain:.1f}s",
                            (cx - 90, 52),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 80), 2)
            else:
                # nose is not yet in the box — prompt the user to look at the camera
                cv2.putText(frame, "CALIBRATING — servo centering on nose",
                            (8, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, calib_rect_col, 1)

        # once calibration is complete, draw a permanent marker at the frame centre
        # so the user can always see where the calibrated origin point is
        elif calib_done:
            cv2.drawMarker(frame, (cx, cy),
                           (0, 255, 220), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(frame, "CAL",
                        (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 220), 1)

        # ── TOP STATUS BAR ────────────────────────────────────────────────────
        # draw a solid black bar across the top of the frame for the status text
        cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
        # choose status text and colour based on current tracking state
        if face_visible:
            status = "TRACKING"; sc = (0, 200, 0)   # green — actively tracking
        elif in_warn:
            status = "WARN";     sc = (0, 80, 255)   # orange — grace period active
        else:
            status = "SCANNING"; sc = (0, 0, 200)    # red — no face found
        # show calibration state: CALIB = done, LOCKING = in progress, NO-CAL = not started
        calib_lbl = "CALIB" if calib_done else ("LOCKING" if in_calibration else "NO-CAL")
        # show number of data points logged if a session is active
        pts = len(active_session["data"]) if active_session else 0
        sess_lbl = f"SESSION  {pts} pts" if active_session else "IDLE"
        # render the full status line across the top of the frame
        cv2.putText(frame,
                    f"PostureBOT  |  {status}  |  {calib_lbl}  |  {sess_lbl}",
                    (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sc, 1)
        # show current FPS in the top-right corner
        cv2.putText(frame, f"FPS:{fps_display:.1f}",
                    (w - 76, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

        # ── LIVE XYZ READOUT ──────────────────────────────────────────────────
        # show live 3D head position in the bottom-left corner during an active session
        if active_session and calib_done:
            lx, ly, lz = latest_xyz
            # dark background box so the text is readable over any frame content
            cv2.rectangle(frame, (0, h - 52), (210, h), (0, 0, 0), -1)
            # display X, Y, Z in cm relative to the calibration origin
            cv2.putText(frame,
                        f"X:{lx:+.1f} Y:{ly:+.1f} Z:{lz:+.1f} cm",
                        (6, h - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
            # display radial distance (straight-line distance from calibration origin)
            cv2.putText(frame,
                        f"r={math.sqrt(lx**2+ly**2+lz**2):.1f} cm",
                        (6, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 180, 255), 1)

        # push the fully annotated frame to the display window
        cv2.imshow("PostureBOT", frame)
        # wait 1ms for a keypress — required for OpenCV to process window events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("e"):
            if tracking_error_data:
                calculate_tracking_error(tracking_error_data)
            else:
                print("[Error] No tracking data collected yet.")

        if key == ord("q"):
            print("[Main] Q pressed — quitting.")
            break

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    # stop the background camera capture thread cleanly
    camera.stop()
    # release the webcam so other applications can use it
    cap.release()
    # close all OpenCV windows
    cv2.destroyAllWindows()
    try:
        # close the serial connection to the ESP32
        ser.close()
    except Exception:
        pass
    print("[Main] Done.")


if __name__ == "__main__":
    main()

