import cv2
import numpy as np

# === 사용자 입력 ===
scene_img_path = './data/scene_img/cg.jpeg'
scene_img = cv2.imread(scene_img_path)
if scene_img is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {scene_img_path}")

# 기준점(씬) 좌표 (3x3)
scene_points = np.array([
    [262, 526], [519, 529], [798, 523],
    [258, 800], [531, 796], [797, 803],
    [250, 1101], [540, 1112], [804, 1104]
], dtype=np.float32)

# 동공 좌표 (3x3)
eye_points = np.array([
    [646, 731], [586, 732], [551, 731],
    [668, 796], [591, 789], [551, 789],
    [655, 860], [587, 846], [523, 843]
], dtype=np.float32)

# 호모그래피 행렬
H = np.array([
    [ 3.48853593e+00, -5.20561166e-01, -2.13046257e+03],
    [-8.35180168e-01, -5.47952129e+00,  4.07909005e+03],
    [-1.38286157e-03, -1.37263945e-03,  1.00000000e+00]
], dtype=np.float64)

# 테스트할 동공 좌표 (실시간 추출값 등)
pupil_center_calib = (280.7, 375.0)

# === 동공 포인트 전체 변환 ===
eye_pts = eye_points.reshape(-1, 1, 2)  # (9, 1, 2)
mapped_eye_pts = cv2.perspectiveTransform(eye_pts, H)  # 결과: (9, 1, 2)

# === 단일 pupil_center 변환 ===
mapped_pupil = None
pt = np.array([[pupil_center_calib]], dtype=np.float32)
mapped_pupil = cv2.perspectiveTransform(pt, H)[0][0]
print(f"pupil_center_calib: {pupil_center_calib} → mapped: {mapped_pupil}")

# === 시각화 ===
debug_img = scene_img.copy()

# 기준점(초록색)
for (x, y) in scene_points:
    cv2.circle(debug_img, (int(x), int(y)), 10, (0, 255, 0), 2)

# 변환된 eye_points (빨간색)
for pt in mapped_eye_pts:
    x, y = int(pt[0][0]), int(pt[0][1])
    cv2.circle(debug_img, (x, y), 10, (0, 0, 255), -1)

# pupil_center 변환 결과 (파란색)
x, y = int(mapped_pupil[0]), int(mapped_pupil[1])
cv2.circle(debug_img, (x, y), 15, (255, 0, 0), 3)
cv2.putText(debug_img, 'pupil_center_calib', (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

cv2.imshow('Homography Debug', debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
