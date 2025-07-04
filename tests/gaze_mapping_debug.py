
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 설명
# 1. 눈 이미지와 씬 이미지를 읽어온다.
# 2. 눈 이미지에서 동공과 반사광을 찾는다.
# 3. 동공과 반사광의 좌표를 이용하여 눈의 시선 벡터를 계산한다.
# 4. 눈의 시선 벡터를 씬 이미지에 매핑한다.
# 5. 매핑된 좌표를 기반으로 씬 이미지에서 눈의 시선 벡터를 계산한다.
# 6. 눈의 시선 벡터를 씬 이미지에 매핑한다.
# 7. 매핑된 좌표를 기반으로 씬 이미지에서 눈의 시선 벡터를 계산한다.
# 8. 눈의 시선 벡터를 씬 이미지에 매핑한다.
eye_img = cv2.imread("./eye_img/sample_eye.jpg")
scene_img = cv2.imread("./scene_img/sample_scene_with_gaze.jpg")

def find_pupil_and_glint_adaptive_debug(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 3000:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx, cy))
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for c in centers:
        cv2.circle(debug_img, c, 3, (0, 255, 0), -1)
    return centers, gray, binary, debug_img

centers, gray, binary, debug_img = find_pupil_and_glint_adaptive_debug(eye_img)
if len(centers) >= 2:
    centers.sort(key=lambda c: c[0]**2 + c[1]**2)
    pupil, glint = centers[-1], centers[0]
else:
    print("Fallback used.")
    pupil, glint = (150, 100), (140, 95)

def get_gaze_vector(pupil, glint):
    dx = pupil[0] - glint[0]
    dy = pupil[1] - glint[1]
    return dx, dy

gaze_vec = get_gaze_vector(pupil, glint)

def map_gaze_to_scene(gaze_vec, scene_size, sensitivity=5):
    center = (scene_size[1] // 2, scene_size[0] // 2)
    return (int(center[0] + sensitivity * gaze_vec[0]), int(center[1] + sensitivity * gaze_vec[1]))

scene_point = map_gaze_to_scene(gaze_vec, scene_img.shape, sensitivity=5)
roi = (500, 150, 620, 320)  # ← 수정!

def get_relative_position(gaze_point, roi):
    x, y = gaze_point
    x1, y1, x2, y2 = roi
    if x1 <= x <= x2 and y1 <= y <= y2:
        return round((x - x1) / (x2 - x1), 3), round((y - y1) / (y2 - y1), 3)
    return None

relative_pos = get_relative_position(scene_point, roi)
cv2.rectangle(scene_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
cv2.circle(scene_img, scene_point, 5, (0, 0, 255), -1)
label = f"ROI gaze: {relative_pos}" if relative_pos else "Gaze outside ROI"
cv2.putText(scene_img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if not relative_pos else (0, 255, 0), 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Eye Grayscale")
axes[1].imshow(binary, cmap="gray")
axes[1].set_title("Thresholded Eye")
axes[2].imshow(cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB))
axes[2].set_title("Scene with Gaze")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
