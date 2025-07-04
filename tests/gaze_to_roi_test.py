import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ==== [사용자 입력 설정] ====
scene_path = './scene_img/scene.JPG'    # 장면 이미지 파일
eye_path = './eye_img/eye.png'        # 눈 이미지 파일

# 그림(ROI) 영역: 전체 scene 이미지 안에서 그림 위치 수동 지정
roi = (200, 100, 400, 300)  # (x1, y1, x2, y2)

# ==== [1단계] 이미지 로드 ====
if not os.path.exists(scene_path) or not os.path.exists(eye_path):
    print("❗ 이미지 파일 경로를 확인하세요.")
    sys.exit(1)

scene_img = cv2.imread(scene_path)
eye_img = cv2.imread(eye_path)

# ==== [2단계] 눈 이미지에서 동공/글린트 검출 ====
def find_pupil_and_glint(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 500:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx, cy))
    if len(centers) >= 2:
        centers.sort(key=lambda c: c[0]**2 + c[1]**2)
        return centers[-1], centers[0]
    return None, None

pupil, glint = find_pupil_and_glint(eye_img)
if pupil is None or glint is None:
    print("❗ 동공 또는 글린트 검출 실패. 입력 영상 확인 필요.")
    sys.exit(1)

# ==== [3단계] 시선 벡터 계산 ====
def get_gaze_vector(pupil, glint):
    dx = pupil[0] - glint[0]
    dy = pupil[1] - glint[1]
    return dx, dy

gaze_vec = get_gaze_vector(pupil, glint)

# ==== [4단계] 시선 벡터를 장면 이미지에 투영 ====
def map_gaze_to_scene(gaze_vec, scene_size, sensitivity=5):
    center = (scene_size[1] // 2, scene_size[0] // 2)
    x = int(center[0] + sensitivity * gaze_vec[0])
    y = int(center[1] + sensitivity * gaze_vec[1])
    return (x, y)

scene_point = map_gaze_to_scene(gaze_vec, scene_img.shape)

# ==== [5단계] ROI 내 상대 좌표 변환 ====
def get_relative_position(gaze_point, roi):
    x, y = gaze_point
    x1, y1, x2, y2 = roi
    if x1 <= x <= x2 and y1 <= y <= y2:
        x_rel = (x - x1) / (x2 - x1)
        y_rel = (y - y1) / (y2 - y1)
        return round(x_rel, 3), round(y_rel, 3)
    return None

relative_pos = get_relative_position(scene_point, roi)

# ==== [6단계] 결과 시각화 ====
cv2.rectangle(scene_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
cv2.circle(scene_img, scene_point, 5, (0, 0, 255), -1)
if relative_pos:
    label = f"Gaze (ROI): ({relative_pos[0]:.2f}, {relative_pos[1]:.2f})"
else:
    label = f"Gaze outside ROI"
cv2.putText(scene_img, label, (10, scene_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

scene_rgb = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
plt.imshow(scene_rgb)
plt.title("Gaze Mapping Result")
plt.axis("off")
plt.show()

# ==== [7단계] 출력 ====
print("✅ Gaze Mapping Summary")
print("Pupil:", pupil)
print("Glint:", glint)
print("Gaze Vector:", gaze_vec)
print("Scene Gaze Point:", scene_point)
print("Relative Position in ROI:", relative_pos if relative_pos else "❌ Outside ROI")
