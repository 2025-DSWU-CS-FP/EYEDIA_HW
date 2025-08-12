import cv2
import numpy as np

# === 영상 불러오기 ===
cap = cv2.VideoCapture("data/eye_video/eye_video.mp4")  # 영상 파일 경로 또는 0 (웹캠)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Grayscale + 대비 강화 (CLAHE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 2️⃣ Adaptive Threshold (동공만 검출)
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 5)

    # 3️⃣ Morphology로 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4️⃣ Contour 탐색
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_center = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 2000:  # 너무 작거나 큰 노이즈 제외
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circularity = 4*np.pi*area/(cv2.arcLength(cnt, True)**2)
        if 0.4 < circularity < 1.2:  # 원형일 경우만
            if area > max_area:  # 가장 큰 원 선택
                pupil_center = (int(x), int(y))
                max_area = area

    # 5️⃣ Contour 결과가 없으면 HoughCircles 시도
    if pupil_center is None:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=50, param1=50, param2=20,
                                   minRadius=5, maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]
            pupil_center = (x, y)

    # 6️⃣ 시각화
    display = frame.copy()
    if pupil_center is not None:
        cv2.circle(display, pupil_center, 5, (0,255,0), 2)  # 중심
        cv2.circle(display, pupil_center, 15, (0,255,0), 1) # 외곽 참고

    cv2.imshow("Eye", display)
    cv2.imshow("Threshold", th)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()