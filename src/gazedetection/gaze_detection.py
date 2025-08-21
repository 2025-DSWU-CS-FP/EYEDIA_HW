import cv2
import dlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib # 모델 저장/불러오기를 위한 라이브러리
import time

# --- 설정 변수 ---
# 모드 설정: 'collect' (데이터 수집) 또는 'predict' (실시간 예측)
MODE = 'collect'

## --- 수정 --- : 화면 구역을 4사분면으로 변경
SCREEN_ZONES = {
    1: "Top-Left", 2: "Top-Right",
    3: "Bot-Left", 4: "Bot-Right"
}
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720 # 예시 화면 크기

## --- 수정 --- : 구역당 수집할 데이터 개수 설정
SAMPLES_PER_ZONE = 20

# Dlib 얼굴 검출기 및 특징점 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- 핵심 기능 함수 (이전 성능 개선 버전 유지) ---

def get_eye_keypoints(shape, gray_frame, eye_points_indices):
    """
    얼굴 특징점에서 눈 영역을 추출하고, 동공과 글린트의 좌표를 계산하는 함수 (성능 개선 버전)
    """
    eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in eye_points_indices], dtype=np.int32)
    
    x, y, w, h = cv2.boundingRect(eye_points)
    if w == 0 or h == 0:
        return None, None, None, None
    eye_roi = gray_frame[y:y+h, x:x+w]
    
    inner_corner = (shape.part(eye_points_indices[3]).x, shape.part(eye_points_indices[3]).y)
    outer_corner = (shape.part(eye_points_indices[0]).x, shape.part(eye_points_indices[0]).y)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eye_roi = clahe.apply(eye_roi)
    
    threshold_eye = cv2.adaptiveThreshold(eye_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    pupil_contour = None
    max_circularity = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area == 0: continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if 0.7 < circularity < 1.2 and 15 < area < 400:
            if circularity > max_circularity:
                max_circularity = circularity
                pupil_contour = c

    pupil_center = None
    if pupil_contour is not None:
        M = cv2.moments(pupil_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00']) + x
            cy = int(M['m01'] / M['m00']) + y
            pupil_center = (cx, cy)
            
    glint_center = None
    if eye_roi.size > 0:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(eye_roi)
        if max_val > 180:
             glint_center = (max_loc[0] + x, max_loc[1] + y)

    return inner_corner, outer_corner, pupil_center, glint_center

def calculate_features(left_eye, right_eye):
    """
    양쪽 눈의 핵심 특징점들을 바탕으로 논문에서 제안한 6가지 특징을 계산하는 함수
    """
    if not all(p is not None for eye in [left_eye, right_eye] for p in eye):
        return None

    l_inner, _, l_pupil, l_glint = left_eye
    r_inner, _, r_pupil, r_glint = right_eye
    
    l_pupil, l_glint, l_inner = np.array(l_pupil), np.array(l_glint), np.array(l_inner)
    r_pupil, r_glint, r_inner = np.array(r_pupil), np.array(r_glint), np.array(r_inner)

    # ... (특징 계산 로직은 이전과 동일)
    vec_l_pg = l_pupil - l_glint
    vec_r_pg = r_pupil - r_glint
    vec_l_pc = l_pupil - l_inner
    vec_r_pc = r_pupil - r_inner
    vec_l_gc = l_glint - l_inner
    vec_r_gc = r_glint - r_inner
    vec_cc = l_inner - r_inner
    dist_cc = np.linalg.norm(vec_cc)
    cos_theta_l = np.dot(vec_l_pg, vec_l_gc) / (np.linalg.norm(vec_l_pg) * np.linalg.norm(vec_l_gc) + 1e-6)
    theta_l = np.arccos(np.clip(cos_theta_l, -1.0, 1.0))
    cos_theta_r = np.dot(vec_r_pg, vec_r_gc) / (np.linalg.norm(vec_r_pg) * np.linalg.norm(vec_r_gc) + 1e-6)
    theta_r = np.arccos(np.clip(cos_theta_r, -1.0, 1.0))
    diff_cc = np.arctan2(vec_cc[1], vec_cc[0])
    feature_vector = np.concatenate([
        vec_l_pg, [np.linalg.norm(vec_l_pg)], vec_r_pg, [np.linalg.norm(vec_r_pg)],
        vec_l_pc, [np.linalg.norm(vec_l_pc)], vec_r_pc, [np.linalg.norm(vec_r_pc)],
        vec_l_gc, [np.linalg.norm(vec_l_gc)], vec_r_gc, [np.linalg.norm(vec_r_gc)],
        [dist_cc, theta_l, theta_r, diff_cc]
    ])
    
    return feature_vector

def draw_screen_zones(frame):
    """화면에 4개 구역과 안내 텍스트를 그리는 함수"""
    ## --- 수정 --- : 2x2 그리드로 변경
    rows, cols = 2, 2
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
    features_data = []
    labels_data = []
    current_zone_to_collect = 1
    ## --- 수정 --- : 각 구역별 수집 카운트를 저장할 딕셔너리
    collected_counts = {i: 0 for i in range(1, 5)}
    
    print("--- 데이터 수집 모드 (자동 진행) ---")
    print(f"각 구역을 응시한 상태에서 '스페이스바'를 눌러 데이터를 {SAMPLES_PER_ZONE}개씩 수집합니다.")
    print("수집이 완료되면 자동으로 다음 구역으로 넘어갑니다.")

elif MODE == 'predict':
    try:
        model = joblib.load('gaze_model.pkl')
        print("--- 실시간 예측 모드 ---")
    except FileNotFoundError:
        print("오류: 학습된 모델 파일(gaze_model.pkl)을 찾을 수 없습니다.")
        exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    display_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    display_frame = draw_screen_zones(display_frame)
    
    features = None  # features 변수 초기화
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        
        left_eye_keypoints = get_eye_keypoints(landmarks, gray, left_eye_indices)
        right_eye_keypoints = get_eye_keypoints(landmarks, gray, right_eye_indices)

        for i in range(36, 48):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        if left_eye_keypoints[2]: cv2.circle(frame, left_eye_keypoints[2], 3, (0, 0, 255), -1)
        if left_eye_keypoints[3]: cv2.circle(frame, left_eye_keypoints[3], 3, (255, 0, 0), -1)
        if right_eye_keypoints[2]: cv2.circle(frame, right_eye_keypoints[2], 3, (0, 0, 255), -1)
        if right_eye_keypoints[3]: cv2.circle(frame, right_eye_keypoints[3], 3, (255, 0, 0), -1)
            
        features = calculate_features(left_eye_keypoints, right_eye_keypoints)

    if MODE == 'collect':
        ## --- 수정 --- : 데이터 수집 UI 및 로직 변경
        if current_zone_to_collect <= 4:
            count = collected_counts[current_zone_to_collect]
            text = f"Look at Zone [{current_zone_to_collect}] ({count}/{SAMPLES_PER_ZONE}). Press SPACE."
            cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            text = "Collection Complete! Press 's' to train and save."
            cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    elif MODE == 'predict' and features is not None:
        prediction = model.predict([features])[0]
        zone_name = SCREEN_ZONES[prediction]
        text = f"Gaze Prediction: Zone {prediction} ({zone_name})"
        cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Gaze Interface", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    if MODE == 'collect':
        ## --- 수정 --- : 스페이스바를 눌렀을 때 데이터 수집 및 자동 진행
        if key == ord(' '): # 스페이스바
            if features is not None and current_zone_to_collect <= 4:
                count = collected_counts[current_zone_to_collect]
                if count < SAMPLES_PER_ZONE:
                    features_data.append(features)
                    labels_data.append(current_zone_to_collect)
                    collected_counts[current_zone_to_collect] += 1
                    print(f"Zone {current_zone_to_collect} data collected. ({collected_counts[current_zone_to_collect]}/{SAMPLES_PER_ZONE})")
                
                # 20개 수집이 완료되면 다음 구역으로 자동 이동
                if collected_counts[current_zone_to_collect] == SAMPLES_PER_ZONE:
                    print(f"Zone {current_zone_to_collect} collection complete!")
                    current_zone_to_collect += 1

        elif key == ord('s'):
            if all(count == SAMPLES_PER_ZONE for count in collected_counts.values()):
                print("\n--- Training Model ---")
                X = np.array(features_data)
                y = np.array(labels_data)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, y)
                
                joblib.dump(model, 'gaze_model.pkl')
                print("Model trained and saved as 'gaze_model.pkl'.")
            else:
                print("Not all zones have enough data. Please complete collection.")

cap.release()
cv2.destroyAllWindows()