import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

EYE_VIDEO_PATH = os.getenv("EYE_VIDEO_PATH")
SCENE_IMAGE_PATH = os.getenv("SCENE_IMAGE_PATH")

cap_eye = cv2.VideoCapture(EYE_VIDEO_PATH)
scene_image = cv2.imread(SCENE_IMAGE_PATH)

if scene_image is None:
    raise FileNotFoundError("SCENE_IMAGE_PATH 경로 확인 필요")

img_h, img_w = scene_image.shape[:2]
canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

# === 동공 검출 함수 ===
def find_pupil_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 블러로 노이즈 제거
    gray_blur = cv2.medianBlur(gray, 5)
    
    # 적응형 이진화
    thresh = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 5
    )
    
    # 노이즈 제거용 모폴로지 연산
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_center = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 2000:  # 동공 크기 범위 필터
            continue
        
        # 원형 정도 측정
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.4:
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        pupil_center = (int(x), int(y))
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        break

    return frame, pupil_center

while cap_eye.isOpened():
    ret, frame = cap_eye.read()
    if not ret:
        break
    
    processed_frame, center = find_pupil_center(frame)
    
    cv2.imshow("Eye", processed_frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap_eye.release()
cv2.destroyAllWindows()
