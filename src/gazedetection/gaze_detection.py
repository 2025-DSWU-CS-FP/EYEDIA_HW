import cv2
import dlib
import numpy as np
from sklearn.svm import SVC
import joblib # 모델 저장/불러오기를 위한 라이브러리
import time

# --- 설정 변수 ---
# 모드 설정: 'collect' (데이터 수집) 또는 'predict' (실시간 예측)
MODE = 'collect'

# 화면 구역 설정 (3x3 grid)
SCREEN_ZONES = {
    1: "Top-Left", 2: "Top-Center", 3: "Top-Right",
    4: "Mid-Left", 5: "Mid-Center", 6: "Mid-Right",
    7: "Bot-Left", 8: "Bot-Center", 9: "Bot-Right"
}
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720 # 예시 화면 크기

# Dlib 얼굴 검출기 및 특징점 예측기 초기화
detector = dlib.get_frontal_face_detector()
# 다운로드한 모델 파일 경로를 정확하게 입력해야 합니다.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- 핵심 기능 함수 ---

def get_eye_keypoints(shape, gray_frame, eye_points_indices):
    """
    얼굴 특징점에서 눈 영역을 추출하고, 동공과 글린트의 좌표를 계산하는 함수
    :param shape: dlib으로 검출된 68개의 얼굴 특징점
    :param gray_frame: 흑백으로 변환된 원본 영상 프레임
    :param eye_points_indices: 왼쪽 눈 또는 오른쪽 눈에 해당하는 특징점 인덱스 리스트
    :return: (눈 안쪽 구석, 눈 바깥쪽 구석, 동공, 글린트) 좌표 튜플
    """
    # 1. 눈 영역의 좌표만 추출
    eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in eye_points_indices], dtype=np.int32)
    
    # 2. 눈 영역만 잘라내기 (Bounding Box)
    x, y, w, h = cv2.boundingRect(eye_points)
    eye_roi = gray_frame[y:y+h, x:x+w]
    
    # 3. 눈 안쪽/바깥쪽 구석 좌표 추출 (논문 기준: inner/outer canthus)
    inner_corner = (shape.part(eye_points_indices[3]).x, shape.part(eye_points_indices[3]).y)
    outer_corner = (shape.part(eye_points_indices[0]).x, shape.part(eye_points_indices[0]).y)
    
    # 4. 동공(Pupil) 검출
    # 이진화를 통해 어두운 영역(동공)을 찾음
    _, threshold_eye = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 윤곽선을 동공으로 간주
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    
    pupil_center = None
    if contours:
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            # 동공의 중심 좌표 계산
            cx = int(M['m10'] / M['m00']) + x
            cy = int(M['m01'] / M['m00']) + y
            pupil_center = (cx, cy)
            
    # 5. 글린트(Glint) 검출
    # 눈 영역에서 가장 밝은 점을 찾음
    glint_center = None
    if eye_roi.size > 0:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(eye_roi)
        if max_val > 200: # 매우 밝은 점이 있을 경우만 글린트로 인정
             glint_center = (max_loc[0] + x, max_loc[1] + y)

    return inner_corner, outer_corner, pupil_center, glint_center

def calculate_features(left_eye, right_eye):
    """
    양쪽 눈의 핵심 특징점들을 바탕으로 논문에서 제안한 6가지 특징을 계산하는 함수
    :param left_eye: 왼쪽 눈의 (inner_corner, outer_corner, pupil, glint) 튜플
    :param right_eye: 오른쪽 눈의 (inner_corner, outer_corner, pupil, glint) 튜플
    :return: 23개의 숫자 데이터로 구성된 특징 벡터 (np.array) 또는 None
    """
    # 필요한 모든 특징점이 검출되었는지 확인
    if not all(p is not None for eye in [left_eye, right_eye] for p in eye):
        return None

    # 각 눈의 특징점 좌표 할당
    l_inner, _, l_pupil, l_glint = left_eye
    r_inner, _, r_pupil, r_glint = right_eye
    
    # 넘파이 배열로 변환하여 계산 용이하게 함
    l_pupil, l_glint, l_inner = np.array(l_pupil), np.array(l_glint), np.array(l_inner)
    r_pupil, r_glint, r_inner = np.array(r_pupil), np.array(r_glint), np.array(r_inner)

    # --- 논문의 6가지 특징 계산 ---
    # 1. 동공 - 글린트 벡터 (Pupil-Glint Vector)
    vec_l_pg = l_pupil - l_glint
    vec_r_pg = r_pupil - r_glint
    
    # 2. 동공 - 눈 안쪽 구석 벡터 (Pupil-Inner Corner Vector)
    vec_l_pc = l_pupil - l_inner
    vec_r_pc = r_pupil - r_inner
    
    # 3. 글린트 - 눈 안쪽 구석 벡터 (Glint-Inner Corner Vector)
    vec_l_gc = l_glint - l_inner
    vec_r_gc = r_glint - r_inner
    
    # 4. 양쪽 눈 안쪽 구석 사이의 거리 벡터
    vec_cc = l_inner - r_inner
    dist_cc = np.linalg.norm(vec_cc)
    
    # 5. 벡터들 사이의 각도 (Theta_PGC)
    # np.dot(a, b) / (norm(a) * norm(b)) = cos(theta)
    cos_theta_l = np.dot(vec_l_pg, vec_l_gc) / (np.linalg.norm(vec_l_pg) * np.linalg.norm(vec_l_gc) + 1e-6)
    theta_l = np.arccos(np.clip(cos_theta_l, -1.0, 1.0))
    
    cos_theta_r = np.dot(vec_r_pg, vec_r_gc) / (np.linalg.norm(vec_r_pg) * np.linalg.norm(vec_r_gc) + 1e-6)
    theta_r = np.arccos(np.clip(cos_theta_r, -1.0, 1.0))
    
    # 6. 편차 각도 (diff_CC)
    diff_cc = np.arctan2(vec_cc[1], vec_cc[0])

    # 모든 특징들을 하나의 벡터로 결합 (총 23개 차원)
    feature_vector = np.concatenate([
        vec_l_pg, [np.linalg.norm(vec_l_pg)],
        vec_r_pg, [np.linalg.norm(vec_r_pg)],
        vec_l_pc, [np.linalg.norm(vec_l_pc)],
        vec_r_pc, [np.linalg.norm(vec_r_pc)],
        vec_l_gc, [np.linalg.norm(vec_l_gc)],
        vec_r_gc, [np.linalg.norm(vec_r_gc)],
        [dist_cc, theta_l, theta_r, diff_cc]
    ])
    
    return feature_vector

def draw_screen_zones(frame):
    """화면에 9개 구역과 안내 텍스트를 그리는 함수"""
    rows, cols = 3, 3
    zone_w, zone_h = SCREEN_WIDTH // cols, SCREEN_HEIGHT // rows
    
    for i in range(1, rows * cols + 1):
        c = (i - 1) % cols
        r = (i - 1) // cols
        x1, y1 = c * zone_w, r * zone_h
        x2, y2 = x1 + zone_w, y1 + zone_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(i), (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# --- 메인 루프 ---

if MODE == 'collect':
    # 데이터 수집 모드
    features_data = []
    labels_data = []
    current_zone_to_collect = 1
    
    print("--- 데이터 수집 모드 ---")
    print("화면에 표시된 숫자를 응시한 후, 키보드에서 해당 숫자 키를 누르세요.")
    print("1번부터 9번까지 순서대로 진행합니다. 각 구역마다 여러 번 눌러 데이터를 수집하세요.")
    print("모든 데이터 수집이 끝나면 's' 키를 눌러 모델을 학습시키고 저장합니다.")

elif MODE == 'predict':
    # 실시간 예측 모드
    try:
        # 학습된 모델 불러오기
        model = joblib.load('gaze_model.pkl')
        print("--- 실시간 예측 모드 ---")
        print("학습된 모델을 성공적으로 불러왔습니다. 시선 추적을 시작합니다.")
    except FileNotFoundError:
        print("오류: 학습된 모델 파일(gaze_model.pkl)을 찾을 수 없습니다.")
        print("먼저 'collect' 모드로 데이터를 수집하고 모델을 학습시켜주세요.")
        exit()

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 화면을 좌우 반전시켜 거울처럼 보이게 함
    frame = cv2.flip(frame, 1)
    
    # 흑백으로 변환하여 처리 속도 향상
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = detector(gray)
    
    # 예측 결과를 표시할 화면 복사본 생성
    display_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    display_frame = draw_screen_zones(display_frame)
    
    for face in faces:
        # 얼굴 특징점 추출
        landmarks = predictor(gray, face)
        
        # 양쪽 눈의 핵심 특징점(구석, 동공, 글린트) 계산
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        
        left_eye_keypoints = get_eye_keypoints(landmarks, gray, left_eye_indices)
        right_eye_keypoints = get_eye_keypoints(landmarks, gray, right_eye_indices)

        # 화면에 검출된 특징점 그리기 (시각화용)
        for i in range(36, 48):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        if left_eye_keypoints[2]: cv2.circle(frame, left_eye_keypoints[2], 3, (0, 0, 255), -1)
        if left_eye_keypoints[3]: cv2.circle(frame, left_eye_keypoints[3], 3, (255, 0, 0), -1)
        if right_eye_keypoints[2]: cv2.circle(frame, right_eye_keypoints[2], 3, (0, 0, 255), -1)
        if right_eye_keypoints[3]: cv2.circle(frame, right_eye_keypoints[3], 3, (255, 0, 0), -1)
            
        # 시선 특징 벡터 계산
        features = calculate_features(left_eye_keypoints, right_eye_keypoints)

        if features is not None:
            if MODE == 'collect':
                # 데이터 수집 안내 메시지
                text = f"[{current_zone_to_collect}] Look at Zone and Press '{current_zone_to_collect}'"
                cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            elif MODE == 'predict':
                # 학습된 모델로 시선 위치 예측
                prediction = model.predict([features])[0]
                zone_name = SCREEN_ZONES[prediction]
                
                # 예측 결과 화면에 표시
                text = f"Gaze Prediction: Zone {prediction} ({zone_name})"
                cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # 화면 보여주기
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Gaze Interface", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        # 'q' 키를 누르면 종료
        break
    
    if MODE == 'collect':
        # 숫자 키가 눌리면 해당 구역의 데이터로 저장
        if ord('1') <= key <= ord('9'):
            zone_num = key - ord('0')
            if features is not None:
                features_data.append(features)
                labels_data.append(zone_num)
                print(f"Zone {zone_num} data collected. Total samples: {len(features_data)}")
                current_zone_to_collect = zone_num + 1 if zone_num < 9 else 9
        
        elif key == ord('s'):
            # 's' 키를 누르면 모델 학습 및 저장
            if len(features_data) > 0:
                print("\n--- Training Model ---")
                X = np.array(features_data)
                y = np.array(labels_data)
                
                # SVM 모델 생성 및 학습
                svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
                svm_model.fit(X, y)
                
                # 모델을 파일로 저장
                joblib.dump(svm_model, 'gaze_model.pkl')
                print("Model trained and saved as 'gaze_model.pkl'.")
                print("You can now switch to 'predict' mode.")
            else:
                print("No data collected. Cannot train model.")

# 종료 처리
cap.release()
cv2.destroyAllWindows()


# 1단계: 데이터 수집 및 모델 학습
# 코드 상단의 MODE 변수를 'collect'로 설정합니다.

# 코드를 실행하면 "Webcam Feed"와 "Gaze Interface" 두 개의 창이 뜹니다.

# "Gaze Interface" 창에 표시된 1번 구역을 응시한 후, 키보드에서 숫자 1 키를 여러 번(최소 10~20번) 눌러 데이터를 수집합니다.

# 같은 방식으로 2번부터 9번까지 모든 구역의 데이터를 수집합니다.

# 모든 구역의 데이터 수집이 끝나면, 키보드에서 s 키를 누릅니다. 그러면 수집된 데이터로 SVM 모델이 학습되고 gaze_model.pkl이라는 파일로 저장됩니다.

# q 키를 눌러 프로그램을 종료합니다.

# 2단계: 실시간 시선 추적
# 코드 상단의 MODE 변수를 'predict'로 수정합니다.

# 코드를 다시 실행합니다.

# 이제 "Gaze Interface" 창에 여러분이 보고 있는 구역이 실시간으로 예측되어 노란색 텍스트로 표시될 것입니다.

