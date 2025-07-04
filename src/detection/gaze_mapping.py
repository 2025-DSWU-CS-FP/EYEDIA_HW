import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일을 읽어서 환경변수로 등록

EYE_VIDEO_PATH = os.getenv("EYE_VIDEO_PATH")
SCENE_IMAGE_PATH = os.getenv("SCENE_IMAGE_PATH")

# === 초기화 ===
cap_eye = cv2.VideoCapture(EYE_VIDEO_PATH)
scene_image = cv2.imread(SCENE_IMAGE_PATH)

# === 호모그래피 행렬 (임시 예시 값) ===
# 실제로는 calibrate.py 등에서 9점 수집 후 생성해야 함
homo_matrix = np.array([
    [1.1, 0.0, 20.0],
    [0.0, 1.1, 10.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# === 함수 정의 ===
def detect_corneal_reflex(gray):
    _, max_val, _, max_loc = cv2.minMaxLoc(gray)
    return max_loc

def remove_corneal_reflex(gray, center, radius=10):
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)
    mean_val = cv2.mean(gray, mask=255 - mask)[0]
    corrected = gray.copy()
    cv2.circle(corrected, center, radius, int(mean_val), -1)
    return corrected

def detect_pupil_contour(corrected, thresh=30):
    _, th = cv2.threshold(corrected, thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h, w = corrected.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 5000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / bh
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 동공 중심이 이미지 중앙에서 너무 멀면 제외
            if abs(cx - w//2) > w//3 or abs(cy - h//2) > h//3:
                continue
            print(f"area: {area}, circularity: {circularity:.2f}, aspect: {aspect_ratio:.2f}, cx: {cx}, cy: {cy}")
            candidates.append(cnt)
    if not candidates:
        return None
    return max(candidates, key=cv2.contourArea)

def fit_ellipse_pupil(contour):
    if contour is None or len(contour) < 5:
        return None
    return cv2.fitEllipse(contour)

def apply_homography(pt, H):
    pt_h = np.array([[pt[0]], [pt[1]], [1]])
    mapped = H @ pt_h
    mapped /= mapped[2]
    return int(mapped[0]), int(mapped[1])

# === 메인 루프 ===
while True:
    ret, frame_eye = cap_eye.read()
    if not ret:
        break

    gray_eye = cv2.cvtColor(frame_eye, cv2.COLOR_BGR2GRAY)
    corneal_center = detect_corneal_reflex(gray_eye)
    corrected_eye = remove_corneal_reflex(gray_eye, corneal_center)
    contour = detect_pupil_contour(corrected_eye)
    ellipse = fit_ellipse_pupil(contour)

    display_scene = scene_image.copy()
    display_eye = cv2.cvtColor(corrected_eye, cv2.COLOR_GRAY2BGR)

    if ellipse:
        pupil_center = ellipse[0]
        cv2.ellipse(display_eye, ellipse, (0, 255, 0), 2)
        gaze_point = apply_homography(pupil_center, homo_matrix)
        cv2.circle(display_scene, gaze_point, 10, (0, 0, 255), -1)
        print(f"동공 위치: {pupil_center}, 시선 좌표: {gaze_point}")

    # 결과 시각화
    cv2.imshow("Eye View", display_eye)
    cv2.imshow("Scene View", display_scene)

    if cv2.waitKey(30) == 27:  # ESC 키
        break

cap_eye.release()
cv2.destroyAllWindows()

def detect_pupil_contour(corrected, thresh=30):
    _, th = cv2.threshold(corrected, thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 2000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.7:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cy = int(M['m01'] / M['m00'])
            if cy < (corrected.shape[0] // 3):
                continue
            candidates.append(cnt)
    if not candidates:
        return None
    return max(candidates, key=cv2.contourArea) 