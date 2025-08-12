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
    [-4.80226959e-01, -1.53920683e+00,  4.79244508e+02],
    [-3.01709050e-01, -2.30403473e+00,  6.39480013e+02],
    [-9.68421382e-04, -3.07336698e-03,  1.00000000e+00]
], dtype=np.float32)

# 개선된 흑백 전처리 및 Hough Circle 적용
def detect_pupil_hough(gray):
    h, w = gray.shape
    roi_x, roi_y, roi_w, roi_h = w//4, h//4, w//2, h//2
    roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # 명암 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_clahe = clahe.apply(roi)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(roi_clahe, (7, 7), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=roi.shape[0]/2,
        param1=50,
        param2=25,
        minRadius=20,
        maxRadius=80
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        pupil = circles[0][0]
        cx, cy, r = pupil
        return (cx + roi_x, cy + roi_y, r)
    return None


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

    pupil = detect_pupil_hough(gray_eye)

    if pupil is not None:
        cx, cy, r = pupil
        cv2.circle(frame_eye, (cx, cy), r, (0, 255, 0), 2)

        real_h, real_w = frame_eye.shape[:2]
        calib_h, calib_w = 1410, 2522

        scale_x = calib_w / real_w
        scale_y = calib_h / real_h

        pupil_center_calib = (cx * scale_x, cy * scale_y)
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