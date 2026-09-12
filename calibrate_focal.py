"""
calibrate_focal.py
-------------------
One-time calibration script to compute a MEASURED focal_px, replacing the
FOV-spec-based estimate in PostureBOT.

HOW TO USE:
  1. Set up a tape measure from your webcam to a chair/mark at a KNOWN
     distance (e.g. 50 cm). Any distance works, as long as it's accurate.
  2. Sit at that exact marked distance, facing the camera straight-on.
  3. Run this script and enter that distance in cm when prompted.
  4. Press SPACE to capture a reading once your face is detected and you're
     holding still at the marked distance. Press ESC to quit.
  5. The script prints a computed focal_px. Copy that number into
     face_tracker_v4.py, replacing the FOV-based focal_px line, e.g.:
         focal_px = <printed_value>   # measured via calibrate_focal.py
     This only needs to be done ONCE -- focal_px is a fixed property of your
     camera/lens, not something that changes per session.

FORMULA (derived from the existing pinhole distance equation):
    z_cm = REAL_FACE_WIDTH_CM * focal_px / face_width_px
  Rearranged to solve for focal_px, using a KNOWN z_cm:
    focal_px = (z_cm * face_width_px) / REAL_FACE_WIDTH_CM
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Constants, matching face_tracker_v4.py ──────────────────────────────
MODEL_PATH         = "face_landmarker.task"
CAM_WIDTH          = 640
CAM_HEIGHT         = 480
REAL_FACE_WIDTH_CM = 13.0   # your ruler-measured face width

LEFT_CHEEK_IDX  = 234
RIGHT_CHEEK_IDX = 454


def main():
    known_z_cm = float(input("Enter your tape-measured distance in cm: "))

    base_opts = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    lm_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(lm_opts)

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    frame_idx = 0
    print("\nHold still at your marked distance. Press SPACE to capture, ESC to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000 / 30))
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_width_px = None
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            h, w = frame.shape[:2]
            face_width_px = abs(lm[RIGHT_CHEEK_IDX].x - lm[LEFT_CHEEK_IDX].x) * w
            cv2.putText(frame, f"face_width_px: {face_width_px:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Target distance: {known_z_cm:.1f} cm  (SPACE=capture, ESC=quit)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Focal Length Calibration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("ESC pressed")
            break

        if key == 32:  # SPACE
            print("SPACE PRESSED")

            if face_width_px is not None:
                print(f"face_width_px = {face_width_px:.2f} px")

                focal_px = (known_z_cm * face_width_px) / REAL_FACE_WIDTH_CM

                print(f"z_cm = {known_z_cm:.1f} cm")
                print(f"focal_px = {focal_px:.2f}")
            else:
                print("SPACE PRESSED, but NO FACE DETECTED")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
