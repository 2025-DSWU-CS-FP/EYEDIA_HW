
import cv2
import numpy as np
# 설명
# 1. 눈 이미지와 씬 이미지를 읽어온다.
# 2. 눈 이미지에서 동공과 반사광을 찾는다.
# 3. 동공과 반사광의 좌표를 이용하여 눈의 시선 벡터를 계산한다.
# 4. 눈의 시선 벡터를 씬 이미지에 매핑한다.
# 5. 매핑된 좌표를 기반으로 씬 이미지에서 눈의 시선 벡터를 계산한다.
# 6. 눈의 시선 벡터를 씬 이미지에 매핑한다.
# 7. 매핑된 좌표를 기반으로 씬 이미지에서 눈의 시선 벡터를 계산한다.
# 8. 눈의 시선 벡터를 씬 이미지에 매핑한다.
video_path = "./eye_video/eye_video.mp4"
cap = cv2.VideoCapture(video_path)
#roi = (500, 150, 620, 320)
roi = (200, 100, 9999, 9999)  # 전체 화면 다 포함

sensitivity = 5

def find_pupil_and_glint(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
    if len(centers) >= 2:
        centers.sort(key=lambda c: c[0]**2 + c[1]**2)
        return centers[-1], centers[0], binary
    return None, None, binary

def get_gaze_vector(pupil, glint):
    dx = pupil[0] - glint[0]
    dy = pupil[1] - glint[1]
    return dx, dy

def map_gaze_to_scene(gaze_vec, size):
    center = (size[1] // 2, size[0] // 2)
    x = int(center[0] + sensitivity * gaze_vec[0])
    y = int(center[1] + sensitivity * gaze_vec[1])
    return (x, y)

def get_relative_position(gaze_point, roi):
    x, y = gaze_point
    x1, y1, x2, y2 = roi
    if x1 <= x <= x2 and y1 <= y <= y2:
        return round((x - x1) / (x2 - x1), 3), round((y - y1) / (y2 - y1), 3)
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pupil, glint, binary = find_pupil_and_glint(frame)
    label = "Detecting..."
    color = (0, 0, 255)

    if pupil and glint:
        gaze_vec = get_gaze_vector(pupil, glint)
        scene_point = map_gaze_to_scene(gaze_vec, frame.shape)
        relative = get_relative_position(scene_point, roi)
        label = f"Gaze: {relative}" if relative else "Outside ROI"
        color = (0, 255, 0) if relative else (0, 0, 255)
        cv2.circle(frame, scene_point, 5, (0, 0, 255), -1)

    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Video Gaze Tracking", frame)

    if cv2.waitKey(20) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
