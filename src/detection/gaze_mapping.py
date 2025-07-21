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

canvas_h, canvas_w = 720, 1280  # 전체 화면 크기
img_x, img_y = 0, 100         # 그림의 좌상단 위치(더 왼쪽으로 이동)
img_h, img_w = scene_image.shape[:2]
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

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
prev_region = None  # 이전 구역 번호 저장 변수
while True:
    ret, frame_eye = cap_eye.read()
    if not ret:
        break
    # 캔버스에 그림 배치
    canvas[:] = 0  # 배경 초기화
    # 그림이 캔버스 밖으로 나가지 않게 범위 체크
    y1, y2 = max(img_y, 0), min(img_y+img_h, canvas_h)
    x1, x2 = max(img_x, 0), min(img_x+img_w, canvas_w)
    sy1, sy2 = y1-img_y, y2-img_y
    sx1, sx2 = x1-img_x, x2-img_x
    canvas[y1:y2, x1:x2] = scene_image[sy1:sy2, sx1:sx2]

    # 그림 위에 4등분 라인 그리기
    cx = img_x + img_w // 2
    cy = img_y + img_h // 2
    cv2.line(canvas, (cx, img_y), (cx, img_y+img_h), (0,255,0), 2)
    cv2.line(canvas, (img_x, cy), (img_x+img_w, cy), (0,255,0), 2)
    # 각 사분면 번호 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, '1', (img_x+20, img_y+40), font, 1.5, (0,255,0), 3)
    cv2.putText(canvas, '2', (cx+20, img_y+40), font, 1.5, (0,255,0), 3)
    cv2.putText(canvas, '3', (img_x+20, cy+40), font, 1.5, (0,255,0), 3)
    cv2.putText(canvas, '4', (cx+20, cy+40), font, 1.5, (0,255,0), 3)

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
        x, y = gaze_point
        if img_x <= x < img_x + img_w and img_y <= y < img_y + img_h:
            rel_x = x - img_x
            rel_y = y - img_y
            if rel_x < img_w // 2 and rel_y < img_h // 2:
                region = 1  # 좌상
            elif rel_x >= img_w // 2 and rel_y < img_h // 2:
                region = 2  # 우상
            elif rel_x < img_w // 2 and rel_y >= img_h // 2:
                region = 3  # 좌하
            else:
                region = 4  # 우하
            print(f"시선이 그림의 {region}영역에 있습니다.")
            # 캔버스에 시선 표시
            cv2.circle(canvas, (x, y), 10, (0, 0, 255), -1)
            # 사분면 텍스트 표시
            cv2.putText(canvas, f"{region}영역", (img_x+img_w//2-80, img_y+img_h+50), font, 1.5, (0,0,255), 3)
        else:
            print("시선이 그림 영역 밖에 있습니다.")

    # 결과 시각화
    cv2.imshow("Eye View", display_eye)
    cv2.imshow("Scene View", canvas)

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