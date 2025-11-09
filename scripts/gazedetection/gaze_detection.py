"""
Organized and refactored gaze detection & data-collection script.
- Maintains original functionality: collect (data collection + train) and demo (real-time prediction)
- Improved logging, clearer structure, better checks, and small debug helpers
- Drop-in replacement: save as gazedetection_organized.py and run with
    python gazedetection_organized.py --mode collect
    python gazedetection_organized.py --mode demo
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import dlib
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Configuration
# -----------------------------
SCREEN_ZONES = {1: "Top-Left", 2: "Top-Right", 3: "Bot-Left", 4: "Bot-Right"}
SAMPLES_PER_ZONE = 20
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720

# Internal lazy globals
_G = {
    "detector": None,
    "predictor": None,
    "model": None,
    "warned": False,
}

# -----------------------------
# Utilities
# -----------------------------

def _resolve_path(filename: str) -> Optional[Path]:
    """Try common locations for model files and return first existing Path or None."""
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [
        cwd / filename,
        here / filename,
        cwd / "data" / "gaze" / filename,
        here / "data" / "gaze" / filename,
    ]
    for p in candidates:
        if p.is_file():
            print(f"✅ Found model file: {p}")
            return p
    print(f"❌ Could not find model file: {filename}")
    return None


def _lazy_init_for_predict() -> bool:
    """Load dlib detector/predictor and trained model on first use.

    Returns True when everything is ready, False otherwise.
    """
    if _G["detector"] is not None and _G["predictor"] is not None and _G["model"] is not None:
        return True

    print("⏳ Initializing dlib models and loading gaze model...")

    try:
        _G["detector"] = dlib.get_frontal_face_detector()
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to init dlib face detector: {e}")
        return False

    sp_path = _resolve_path("shape_predictor_68_face_landmarks.dat")
    if sp_path is None:
        if not _G["warned"]:
            print("ℹ️ Predictor (.dat) not found -> gaze disabled.")
            _G["warned"] = True
        return False
    try:
        _G["predictor"] = dlib.shape_predictor(str(sp_path))
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load shape predictor '{sp_path}': {e}")
        return False

    gm_path = _resolve_path("gaze_model.pkl")
    if gm_path is None:
        if not _G["warned"]:
            print("ℹ️ gaze_model.pkl not found -> gaze disabled.")
            _G["warned"] = True
        return False
    try:
        _G["model"] = joblib.load(str(gm_path))
        print("✅ Gaze model loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load gaze model '{gm_path}': {e}")
        return False

    return True


# -----------------------------
# Eye/keypoint extraction & feature calc
# -----------------------------

def _get_eye_keypoints(shape: dlib.full_object_detection, gray_frame: np.ndarray, eye_points_indices) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Return (inner_corner, outer_corner, pupil_center, glint_center) or (None,..) on failure."""
    try:
        eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in eye_points_indices], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(eye_points)
        if w <= 0 or h <= 0:
            return None, None, None, None

        eye_roi = gray_frame[y:y + h, x:x + w]
        if eye_roi.size == 0:
            return None, None, None, None

        inner_corner = (shape.part(eye_points_indices[3]).x, shape.part(eye_points_indices[3]).y)
        outer_corner = (shape.part(eye_points_indices[0]).x, shape.part(eye_points_indices[0]).y)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eye_roi_clahe = clahe.apply(eye_roi)

        thr = cv2.adaptiveThreshold(eye_roi_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        pupil_contour = None
        max_circ = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 15:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circ = 4 * np.pi * (area / (peri * peri))
            if 0.6 < circ < 1.3 and area < (w * h * 0.5):
                if circ > max_circ:
                    max_circ = circ
                    pupil_contour = c

        pupil_center = None
        if pupil_contour is not None:
            M = cv2.moments(pupil_contour)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"]) + x
                cy = int(M["m01"] / M["m00"]) + y
                pupil_center = (cx, cy)

        glint_center = None
        if eye_roi_clahe.size > 0:
            _, max_val, _, max_loc = cv2.minMaxLoc(eye_roi_clahe)
            if max_val > 150:
                glint_center = (max_loc[0] + x, max_loc[1] + y)

        return inner_corner, outer_corner, pupil_center, glint_center
    except Exception:
        return None, None, None, None


def _calc_features(left_eye, right_eye) -> Optional[np.ndarray]:
    """Compute features using only robust quantities (no explicit glint dependence).

    Returns 1D numpy array or None.
    """
    if left_eye is None or right_eye is None:
        return None

    # require inner corner and pupil center at minimum
    if left_eye[0] is None or left_eye[2] is None or right_eye[0] is None or right_eye[2] is None:
        return None

    l_inner, _, l_pupil, _ = left_eye
    r_inner, _, r_pupil, _ = right_eye

    try:
        l_pupil = np.array(l_pupil, dtype=float)
        r_pupil = np.array(r_pupil, dtype=float)
        l_inner = np.array(l_inner, dtype=float)
        r_inner = np.array(r_inner, dtype=float)
    except Exception:
        return None

    vl_pc = l_pupil - l_inner
    vr_pc = r_pupil - r_inner
    vcc = l_inner - r_inner

    def L(v):
        return np.linalg.norm(v)

    dist_cc = L(vcc)
    diff_cc = np.arctan2(vcc[1], vcc[0])

    feats = np.concatenate([
        vl_pc, [L(vl_pc)],
        vr_pc, [L(vr_pc)],
        [dist_cc, diff_cc]
    ])

    if not np.all(np.isfinite(feats)):
        return None

    return feats


# -----------------------------
# Public API: predict_zone
# -----------------------------

def predict_zone(frame_bgr: np.ndarray) -> Optional[int]:
    """Predict which screen zone the user is looking at (1..4) or None on failure."""
    if not _lazy_init_for_predict():
        return None

    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"Error converting to grayscale: {e}")
        return None

    faces = _G["detector"](gray)
    if len(faces) == 0:
        return None

    face = faces[0]
    try:
        landmarks = _G["predictor"](gray, face)
    except Exception as e:
        print(f"Error predicting landmarks: {e}")
        return None

    left_eye = _get_eye_keypoints(landmarks, gray, list(range(36, 42)))
    right_eye = _get_eye_keypoints(landmarks, gray, list(range(42, 48)))
    feats = _calc_features(left_eye, right_eye)
    if feats is None:
        return None

    try:
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        zone = int(_G["model"].predict(feats)[0])
        if zone in SCREEN_ZONES:
            return zone
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

    return None


# -----------------------------
# Visualization helpers
# -----------------------------

def draw_screen_zones(frame: np.ndarray) -> np.ndarray:
    rows, cols = 2, 2
    zone_w, zone_h = SCREEN_WIDTH // cols, SCREEN_HEIGHT // rows
    for i in range(1, rows * cols + 1):
        c = (i - 1) % cols
        r = (i - 1) // cols
        x1, y1 = c * zone_w, r * zone_h
        x2, y2 = x1 + zone_w, y1 + zone_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(i), (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


# -----------------------------
# Collect & Train
# -----------------------------

def _run_collect_and_train():
    print("--- Data collection & training mode ---")
    print(f"Each zone: press SPACE to capture; target {SAMPLES_PER_ZONE} samples per zone.")

    try:
        local_detector = dlib.get_frontal_face_detector()
        sp_path = _resolve_path("shape_predictor_68_face_landmarks.dat")
        if sp_path is None:
            print("❌ predictor(.dat) not found. Aborting.")
            return
        local_predictor = dlib.shape_predictor(str(sp_path))
    except Exception as e:
        print(f"❌ Failed to prepare dlib models: {e}")
        return

    features_data, labels_data = [], []
    current_zone_to_collect = 1
    collected_counts = {i: 0 for i in SCREEN_ZONES.keys()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to open camera. Check index/permission/occupied apps.")
        return

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            features = None
            debug_face_box = None

            faces = local_detector(gray)
            if faces:
                face = faces[0]
                debug_face_box = (face.left(), face.top(), face.right(), face.bottom())
                try:
                    landmarks = local_predictor(gray, face)
                    left_eye = _get_eye_keypoints(landmarks, gray, list(range(36, 42)))
                    right_eye = _get_eye_keypoints(landmarks, gray, list(range(42, 48)))
                    features = _calc_features(left_eye, right_eye)
                except Exception as e:
                    print(f"Warning: feature extraction error: {e}")
                    features = None

            disp = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            disp = draw_screen_zones(disp)

            if current_zone_to_collect <= len(SCREEN_ZONES):
                cnt = collected_counts[current_zone_to_collect]
                text = f"Look at Zone [{current_zone_to_collect}] ({cnt}/{SAMPLES_PER_ZONE}). Press SPACE."
                color = (255, 255, 255)
                if features is None:
                    text += " (Face/Eyes Not Detected!)"
                    color = (0, 0, 255)
            else:
                text = "Collection Complete! Press 's' to train and save."
                color = (0, 255, 255)

            cv2.putText(disp, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Gaze Training Instructions", disp)

            if debug_face_box:
                cv2.rectangle(frame, (debug_face_box[0], debug_face_box[1]), (debug_face_box[2], debug_face_box[3]), (0, 255, 0), 2)
            feat_status = f"Features: {'OK' if features is not None else 'None'}"
            cv2.putText(frame, feat_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Camera Feed (for Debug)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Collection aborted by user.")
                break

            if key == ord(' ') and features is not None and current_zone_to_collect <= len(SCREEN_ZONES):
                if collected_counts[current_zone_to_collect] < SAMPLES_PER_ZONE:
                    features_data.append(features)
                    labels_data.append(current_zone_to_collect)
                    collected_counts[current_zone_to_collect] += 1
                    print(f"Zone {current_zone_to_collect} collected: {collected_counts[current_zone_to_collect]}/{SAMPLES_PER_ZONE}")

                if collected_counts[current_zone_to_collect] == SAMPLES_PER_ZONE:
                    print(f"✅ Zone {current_zone_to_collect} collection finished.")
                    current_zone_to_collect += 1
                    if current_zone_to_collect > len(SCREEN_ZONES):
                        print("All zones collected. Press 's' to train.")

            elif key == ord('s'):
                if all(v >= SAMPLES_PER_ZONE for v in collected_counts.values()):
                    print("--- Training model ---")
                    X = np.array(features_data)
                    y = np.array(labels_data)
                    try:
                        model_to_save = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        model_to_save.fit(X, y)
                    except Exception as e:
                        print(f"❌ Training failed: {e}")
                        continue

                    save_path = Path.cwd() / "gaze_model.pkl"
                    try:
                        joblib.dump(model_to_save, save_path)
                        print(f"✅ Model saved: {save_path}")
                        time.sleep(2)
                        break
                    except Exception as e:
                        print(f"❌ Failed to save model: {e}")
                        break
                else:
                    print("⚠️ Not enough data for all zones:", collected_counts)

            frame_count += 1

    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")


# -----------------------------
# Demo
# -----------------------------

def _run_predict_demo():
    print("--- Live prediction demo ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to open camera for demo.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break

            frame = cv2.flip(frame, 1)
            zone = predict_zone(frame)
            text = f"Gaze Prediction: Zone {zone}" if zone else "Gaze Prediction: (Detecting...)"
            color = (0, 255, 255) if zone else (0, 0, 255)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # draw face box if detector available
            if _G.get("detector") is not None:
                gray_demo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_demo = _G["detector"](gray_demo)
                if faces_demo:
                    fd = faces_demo[0]
                    cv2.rectangle(frame, (fd.left(), fd.top()), (fd.right(), fd.bottom()), (0, 255, 0), 2)

            cv2.imshow("EYEDIA Gaze Demo", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Demo ended.")


# -----------------------------
# CLI entry
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EYEDIA Gaze Detection Module")
    parser.add_argument("--mode", choices=["collect", "demo"], default="demo", help="collect or demo")
    args = parser.parse_args()

    if args.mode == "collect":
        _run_collect_and_train()
    else:
        _run_predict_demo()
