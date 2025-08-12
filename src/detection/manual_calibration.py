import cv2
import numpy as np
import sys
import os

# === 1. 눈 이미지에서 동공 중심 좌표 수동 추출 ===
EYE_IMG_DIR = './data/calibration/v2'
CALIB_WIDTH = 1024  # calibration 이미지 너비 (좌우 반전을 위해 필요)

eye_points = []

def click_eye(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"동공 클릭 좌표: ({x}, {y})")
        eye_points.append([x, y])

img_paths = [
    os.path.join(EYE_IMG_DIR, f'[{i}][{j}].png') for i in range(3) for j in range(3)
]

for idx, path in enumerate(img_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {path}")
        sys.exit(1)
    cv2.imshow('eye', img)
    cv2.setMouseCallback('eye', click_eye)
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("ESC를 눌러 종료합니다.")
            cv2.destroyAllWindows()
            sys.exit(0)
        if len(eye_points) == idx + 1:
            cv2.destroyAllWindows()
            break

print("\n원본 eye_points =", eye_points)

# === 2. 기준점(씬) 이미지에서 9개 점 중심 좌표 수동 추출 ===
scene_points = []

def click_scene(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"기준점 클릭 좌표: ({x}, {y})")
        scene_points.append([x, y])

scene_img_path = './data/scene_img/cg.jpeg'
scene_img = cv2.imread(scene_img_path)
if scene_img is None:
    print(f"이미지를 불러올 수 없습니다: {scene_img_path}")
    sys.exit(1)

scene_h, scene_w = scene_img.shape[:2]
print(f"\n씬 이미지 크기: {scene_w}x{scene_h}")

cv2.imshow('scene', scene_img)
cv2.setMouseCallback('scene', click_scene)
while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        print("ESC를 눌러 종료합니다.")
        cv2.destroyAllWindows()
        sys.exit(0)
    if len(scene_points) == 9:
        cv2.destroyAllWindows()
        break

print("scene_points =", scene_points)

# === 3. 호모그래피 계산 ===

# 좌우 반전된 eye 좌표 생성
eye_points_flipped = [[CALIB_WIDTH - x, y] for x, y in eye_points]

eye_points_np = np.array(eye_points_flipped, dtype=np.float32)
scene_points_np = np.array(scene_points, dtype=np.float32)

H, status = cv2.findHomography(eye_points_np, scene_points_np)

print("\n좌우 반전된 eye_points =", eye_points_flipped)
print("\n호모그래피 행렬 (좌우 반전 보정 포함):\n", H)

# 선택: 저장
np.save('./data/homography_matrix.npy', H)
print("\n✅ homography_matrix.npy 저장 완료!")
