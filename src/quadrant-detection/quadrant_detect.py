import cv2
import numpy as np
from collections import deque

#VIDEO_PATH = "data/eye_video/eye_video.mp4"
VIDEO_PATH = "data/eye_video/25.08.02yooni.mp4"
OUTPUT_PATH = "./output_gaze.avi"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

history = deque(maxlen=7)  # 최근 7프레임 평균

# 초기 기준선 (화면 중앙)
baseline_x = width // 2
baseline_y = height // 2
baseline_frames = 30  # 초기 30프레임동안 baseline 업데이트

frame_count = 0

# def detect_pupil_circle(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#     gray = cv2.medianBlur(gray, 5)

#     # 동공 후보 탐색
#     circles = cv2.HoughCircles(
#         gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
#         param1=60, param2=20, minRadius=5, maxRadius=40
#     )

#     # 동공이 검출되면 가장 큰 원을 선택
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         # 가장 큰 원(동공일 확률 높은 것) 선택
#         largest = max(circles[0, :], key=lambda x: x[2])
#         cx, cy, r = largest
#         return int(cx), int(cy), int(r)
#     return None
def detect_pupil_circle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape

    # 1. ROI 제한 (윗부분 제거 - 속눈썹 방지)
    roi_y1 = int(h * 0.2)  # 위쪽 40% 잘라냄
    roi = gray[roi_y1:h, :]

    # 2. 어두운 부분만 남김 (밝은 피부 제거)
    _, mask = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
    roi_masked = cv2.bitwise_and(roi, mask)

    # 3. Hough 원 검출
    circles = cv2.HoughCircles(
        roi_masked, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=60, param2=20, minRadius=8, maxRadius=40
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # 위치 필터: 중앙 근처 원만
        valid = []
        for x, y, r in circles[0, :]:
            global_y = y + roi_y1  # ROI offset 보정
            if abs(global_y - h/2) < h*0.3 and abs(x - w/2) < w*0.3:
                valid.append((x, global_y, r))

        if valid:
            # 가장 큰 원 선택
            cx, cy, r = max(valid, key=lambda c: c[2])
            return int(cx), int(cy), int(r)
    return None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    pupil = detect_pupil_circle(frame)

    if pupil:
        cx, cy, r = pupil

        # 초기 baseline 업데이트
        if frame_count <= baseline_frames:
            baseline_x = int(0.9 * baseline_x + 0.1 * cx)
            baseline_y = int(0.9 * baseline_y + 0.1 * cy)

        # 사분면 판정
        horiz = "L" if cx < baseline_x else "R"
        vert = "T" if cy < baseline_y else "B"
        quadrant = vert + horiz

        history.append(quadrant)
        final_quadrant = max(set(history), key=history.count)

        # 동공 표시
        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(frame, f"Gaze: {final_quadrant}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Eye Detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
