import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

EYE_VIDEO_PATH = os.getenv("EYE_VIDEO_PATH")
SCENE_IMAGE_PATH = os.getenv("SCENE_IMAGE_PATH")

cap_eye = cv2.VideoCapture(EYE_VIDEO_PATH)
scene_image = cv2.imread(SCENE_IMAGE_PATH)

img_h, img_w = scene_image.shape[:2]
canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

homo_matrix = np.array([
    [-7.71402754e-01,  3.06218830e-01,  4.98551619e+02],
    [ 2.80580863e+00, -2.17511039e+00,  7.63113998e+02],
    [ 2.73472789e-03, -3.17424936e-03,  1.00000000e+00]
], dtype=np.float32)

# 개선된 동공 검출
def detect_pupil_contour(corrected, thresh=15):
    blurred = cv2.GaussianBlur(corrected, (9, 9), 0)
    _, th = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h, w = corrected.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 8000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.4:
            continue

        ellipse = None
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (MA, ma), angle = ellipse

            aspect_ratio = max(MA, ma) / min(MA, ma)
            if aspect_ratio > 2.5:
                continue

            if abs(cx - w//2) > w//4 or abs(cy - h//2) > h//4:
                continue

            candidates.append(cnt)

    return max(candidates, key=cv2.contourArea) if candidates else None


def apply_homography(pt, H):
    pt_h = np.array([[pt[0]], [pt[1]], [1]])
    mapped = H @ pt_h
    mapped /= mapped[2]
    return int(mapped[0]), int(mapped[1])


while True:
    ret, frame_eye = cap_eye.read()
    if not ret:
        break

    canvas[:] = scene_image.copy()

    gray_eye = cv2.cvtColor(frame_eye, cv2.COLOR_BGR2GRAY)

    contour = detect_pupil_contour(gray_eye)
    ellipse = cv2.fitEllipse(contour) if contour is not None else None

    if ellipse:
        pupil_center = ellipse[0]
        cv2.ellipse(frame_eye, ellipse, (0, 255, 0), 2)

        real_h, real_w = frame_eye.shape[:2]
        calib_h, calib_w = 1410, 2522

        scale_x = calib_w / real_w
        scale_y = calib_h / real_h

        pupil_center_calib = (pupil_center[0] * scale_x, pupil_center[1] * scale_y)
        gaze_point = apply_homography(pupil_center_calib, homo_matrix)

        x, y = gaze_point
        if 0 <= x < img_w and 0 <= y < img_h:
            cv2.circle(canvas, (x, y), 10, (0, 0, 255), -1)
            region = 1 + (x >= img_w // 2) + 2 * (y >= img_h // 2)
            cv2.putText(canvas, f"Region: {region}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Eye", frame_eye)
    cv2.imshow("Scene", canvas)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap_eye.release()
cv2.destroyAllWindows()
