# # import cv2
# # import numpy as np
# # import os
# # import sys

# # # === 설정 ===
# # EYE_IMG_DIR = './data/caligration/v2'
# # SCENE_IMG_PATH = './data/scene_img/cg.jpeg'
# # img_paths = [os.path.join(EYE_IMG_DIR, f'[{i}][{j}].png') for i in range(3) for j in range(3)]

# # # === 동공 중심 자동 검출 함수 ===
# # def find_pupil_center(img):
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.medianBlur(gray, 5)

# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# #     gray = clahe.apply(gray)

# #     th = cv2.adaptiveThreshold(gray, 255,
# #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
# #         11, 5)

# #     contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     candidates = []
# #     for cnt in contours:
# #         area = cv2.contourArea(cnt)
# #         if 100 < area < 5000:
# #             perimeter = cv2.arcLength(cnt, True)
# #             if perimeter == 0:
# #                 continue
# #             circularity = 4 * np.pi * area / (perimeter * perimeter)
# #             if circularity < 0.4:
# #                 continue
# #             candidates.append(cnt)

# #     if not candidates:
# #         return None

# #     cnt = max(candidates, key=cv2.contourArea)
# #     M = cv2.moments(cnt)
# #     if M['m00'] == 0:
# #         return None
# #     cx = int(M['m10'] / M['m00'])
# #     cy = int(M['m01'] / M['m00'])
# #     return (cx, cy)

# # # === 수동 클릭 함수 ===
# # def manual_select_center(img, title):
# #     selected = []
# #     def on_mouse(event, x, y, flags, param):
# #         if event == cv2.EVENT_LBUTTONDOWN:
# #             print(f"[수동 입력] ({x}, {y})")
# #             selected.append((x, y))
# #     cv2.imshow(title, img)
# #     cv2.setMouseCallback(title, on_mouse)
# #     while len(selected) < 1:
# #         if cv2.waitKey(1) == 27:
# #             print("ESC 입력으로 종료됨")
# #             sys.exit(0)
# #     cv2.destroyWindow(title)
# #     return selected[0]

# # # === 1. 눈 이미지에서 동공 중심 추출 ===
# # eye_points = []
# # for path in img_paths:
# #     img = cv2.imread(path)
# #     if img is None:
# #         print(f"이미지를 불러올 수 없습니다: {path}")
# #         sys.exit(1)

# #     center = find_pupil_center(img)
# #     if center is None:
# #         print(f"⚠️ 자동 검출 실패 → 클릭으로 지정: {path}")
# #         center = manual_select_center(img, f"Click Pupil Center: {os.path.basename(path)}")
# #     else:
# #         print(f"[자동 검출] {path} → {center}")

# #     eye_points.append(center)

# # print("\n✅ 동공 중심 좌표 추출 완료")
# # print("eye_points =", eye_points)

# # # === 2. 씬 이미지에서 9개 기준점 수집 ===
# # scene_points = []
# # def click_scene(event, x, y, flags, param):
# #     if event == cv2.EVENT_LBUTTONDOWN:
# #         print(f"[기준점 클릭] ({x}, {y})")
# #         scene_points.append([x, y])

# # scene_img = cv2.imread(SCENE_IMG_PATH)
# # if scene_img is None:
# #     print(f"씬 이미지를 불러올 수 없습니다: {SCENE_IMG_PATH}")
# #     sys.exit(1)

# # cv2.imshow('Scene Image - Click 9 points', scene_img)
# # cv2.setMouseCallback('Scene Image - Click 9 points', click_scene)

# # while len(scene_points) < 9:
# #     if cv2.waitKey(1) == 27:
# #         print("ESC 입력으로 종료됨")
# #         sys.exit(0)

# # cv2.destroyAllWindows()

# # print("\n✅ 씬 이미지 기준점 수집 완료")
# # print("scene_points =", scene_points)

# # # === 3. 호모그래피 계산 ===
# # eye_points_np = np.array(eye_points, dtype=np.float32)
# # scene_points_np = np.array(scene_points, dtype=np.float32)
# # H, status = cv2.findHomography(eye_points_np, scene_points_np)

# # print("\n✅ 호모그래피 행렬:")
# # print(H)import cv2import cv2import cv2
# import numpy as np
# import cv2

# VIDEO_PATH = "./data/eye_video/25.08.02yooni_rasp.mp4"
# # 9개 씬 기준점 (예시: 네가 클릭해서 얻은 좌표)
# SCENE_POINTS = [
#     (255, 538), (518, 534), (797, 527),
#     (257, 803), (538, 807), (808, 800),
#     (254, 1107), (545, 1111), (812, 1099)
# ]

# # 각 점별 원래 시작·종료 시간 (초 단위)
# TIME_RANGES = [
#     (1.7, 5.0),
#     (6.3, 7.0),
#     (8.0, 8.7),

#     (9.5, 10.5),
#     (10.9, 11.5),
#     (11.8, 12.5),

#     (13.0, 14.0),
#     (14.7, 15.3),
#     (15.7, 17.2)
# ]

# SLOW_FACTOR = 0.5  # 0.5배속

# # --- 동공 및 각막반사 검출 ---
# def detect_pupil_and_cr(gray):
#     # 동공: 단순 threshold → contour
#     _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return None, None

#     cnt = max(contours, key=cv2.contourArea)
#     (px, py), pr = cv2.minEnclosingCircle(cnt)
#     pupil_center = (int(px), int(py))

#     # 각막반사: 가장 밝은 점
#     _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
#     return pupil_center, maxLoc

# # --- 비디오 열기 ---
# cap = cv2.VideoCapture(VIDEO_PATH)
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Video FPS: {fps}")

# eye_points = []
# scene_points = []

# for idx, ((sx, sy), (start_t, end_t)) in enumerate(zip(SCENE_POINTS, TIME_RANGES)):
#     # 0.5배속 처리 → 실제 구간을 2배로 확장
#     adj_start = start_t
#     adj_end = start_t + (end_t - start_t) / SLOW_FACTOR

#     start_f = int(adj_start * fps)
#     end_f = int(adj_end * fps)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

#     pupil_list = []
#     cr_list = []

#     print(f"=== {idx+1}/9 점 ({sx}, {sy}) ===")

#     for f in range(start_f, end_f):
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ 영상 끝")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         pupil, cr = detect_pupil_and_cr(gray)
#         if pupil and cr:
#             pupil_list.append(pupil)
#             cr_list.append(cr)

#     if pupil_list:
#         pupil_avg = np.mean(pupil_list, axis=0)
#         cr_avg = np.mean(cr_list, axis=0)
#         eye_vec = pupil_avg - cr_avg
#         eye_points.append(eye_vec)
#         scene_points.append([sx, sy])
#         print(f"캘리브레이션 점 기록: Eye={eye_vec}, Scene=({sx}, {sy})")
#     else:
#         print("⚠️ 검출 실패, skip")
#         eye_points.append([0, 0])
#         scene_points.append([sx, sy])

# cap.release()

# # --- 호모그래피 계산 ---
# eye_points_np = np.array(eye_points, dtype=np.float32)
# scene_points_np = np.array(scene_points, dtype=np.float32)
# H, status = cv2.findHomography(eye_points_np, scene_points_np)

# print("\n✅ 호모그래피 행렬")
# print(H)
import numpy as np
import cv2
import os
import sys

VIDEO_PATH = "./data/eye_video/25.08.02yooni_rasp.mp4"

# 9개 씬 기준점 (예시)
SCENE_POINTS = [
    (255, 538), (518, 534), (797, 527),
    (257, 803), (538, 807), (808, 800),
    (254, 1107), (545, 1111), (812, 1099)
]

TIME_RANGES = [
    (0.0, 0.5),
    (0.5, 2.0),
    (2.8, 4.0),

    (5.0, 6.7),
    (7.0, 7.9),
    (8.9, 10.5),

    (11.0, 12.2),
    (12.8, 13.2),
    (13.8, 14.4)
]

SLOW_FACTOR = 0.5  # 0.5배속

# --- 동공 및 각막반사 검출 ---
def detect_pupil_and_cr(gray):
    # 동공: 단순 threshold → contour
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    (px, py), pr = cv2.minEnclosingCircle(cnt)
    pupil_center = (int(px), int(py))

    # 각막반사: 가장 밝은 점
    _, _, _, maxLoc = cv2.minMaxLoc(gray)
    return pupil_center, maxLoc

# --- 비디오 열기 ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

eye_points = []
scene_points = []

for idx, ((sx, sy), (start_t, end_t)) in enumerate(zip(SCENE_POINTS, TIME_RANGES)):
    adj_start = start_t
    adj_end = start_t + (end_t - start_t) / SLOW_FACTOR

    start_f = int(adj_start * fps)
    end_f = int(adj_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    pupil_list = []
    cr_list = []

    print(f"=== {idx+1}/9 점 ({sx}, {sy}) ===")

    for f in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 영상 끝")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil, cr = detect_pupil_and_cr(gray)

        if pupil and cr:
            pupil_list.append(pupil)
            cr_list.append(cr)

            # --- 시각화 ---
            vis = frame.copy()
            # cv2.circle(vis, pupil, 5, (0, 255, 0), -1)  # 동공: 초록색
            # cv2.circle(vis, cr, 3, (0, 0, 255), -1)     # 각막반사: 빨간색
            cv2.circle(vis, pupil, 15, (0, 255, 0), -1)  # 동공: 초록색, 더 크게
            cv2.circle(vis, cr, 8, (0, 0, 255), -1)      # 각막반사: 빨간색, 더 크게

            cv2.imshow("Eye Detection", vis)

        # q 누르면 중간 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
cv2.destroyAllWindows()

# --- 호모그래피 계산 ---
eye_points_np = np.array(eye_points, dtype=np.float32)
scene_points_np = np.array(scene_points, dtype=np.float32)
H, status = cv2.findHomography(eye_points_np, scene_points_np)

print("\n✅ 호모그래피 행렬")
print(H)

# --- np.array 저장 ---
SAVE_DIR = './data/calibration_results'
np.save(os.path.join(SAVE_DIR, 'eye_points.npy'), eye_points_np)
np.save(os.path.join(SAVE_DIR, 'scene_points.npy'), scene_points_np)
np.save(os.path.join(SAVE_DIR, 'homography.npy'), H)
print(f"\n✅ 저장 완료: {SAVE_DIR}")