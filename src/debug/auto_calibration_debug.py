import cv2
import numpy as np

VIDEO_PATH = "./data/eye_video/25.07.21yooni.mp4"

# 기준 Scene 포인트 (9점)
SCENE_POINTS = [
    (255, 538), (518, 534), (797, 527),
    (257, 803), (538, 807), (808, 800),
    (254, 1107), (545, 1111), (812, 1099)
]

# 각 점별 시간 구간 (초 단위)
TIME_RANGES = [
    (0.0, 0.9),
    (1.0, 3.0),
    (3.1, 4.0),
    (5.0, 5.9),
    (6.0, 6.9),
    (7.0, 7.9),
    (9.0, 10.0),
    (11.0, 11.9),
    (12.0, 12.9)
]

def detect_pupil_and_cr(gray):
    """동공과 각막반사 위치 추출"""
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    (px, py), pr = cv2.minEnclosingCircle(cnt)
    pupil_center = (int(px), int(py))

    _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
    return pupil_center, maxLoc

# --- 비디오 열기 ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

eye_points = []
scene_points = []

for idx, ((sx, sy), (start_t, end_t)) in enumerate(zip(SCENE_POINTS, TIME_RANGES)):
    start_f = int(start_t * fps)
    end_f = int(end_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    pupil_list = []
    cr_list = []

    print(f"=== {idx+1}/9 점 ({sx}, {sy}) ===")

    for f in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil, cr = detect_pupil_and_cr(gray)
        if pupil and cr:
            pupil_list.append(pupil)
            cr_list.append(cr)

    if pupil_list:
        pupil_avg = np.mean(pupil_list, axis=0)
        cr_avg = np.mean(cr_list, axis=0)
        eye_vec = pupil_avg - cr_avg
        eye_points.append(eye_vec)
        scene_points.append([sx, sy])
        print(f"캘리브레이션 점 기록: Eye={eye_vec}, Scene=({sx}, {sy})")
    else:
        print("⚠️ 검출 실패, skip")
        eye_points.append([0, 0])
        scene_points.append([sx, sy])

cap.release()

# --- 호모그래피 계산 ---
eye_points_np = np.array(eye_points, dtype=np.float32)
scene_points_np = np.array(scene_points, dtype=np.float32)
H, status = cv2.findHomography(eye_points_np, scene_points_np)

print("\n✅ 호모그래피 행렬")
print(H)

# --- 디버그: 예시 pupil -> scene 변환 ---
print("\n[디버그] 랜덤 pupil 좌표 변환 예시:")
test_pupil = np.array([[[280, 375]]], dtype=np.float32)  # 예시 동공 좌표
mapped_pt = cv2.perspectiveTransform(test_pupil, H)[0][0]
print(f"pupil {test_pupil[0][0]} -> scene {mapped_pt.astype(int)}")
