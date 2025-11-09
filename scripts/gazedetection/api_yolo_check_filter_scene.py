#!/usr/bin/env python3
# eyedia_dualcam_remote_gaze.py
# 듀얼 카메라(Scene, Eye)와 리모컨 입력을 사용하는 스크립트
#
# [키 매핑]
# - TAB   → detect-art    → POST /process-image?art_id=...
# - ENTER → detect-area   → POST /process-image?art_id=...&q=Qn (Gaze model)
# - 1~4   → Q1~Q4 수동 변경 (Gaze 모델 테스트 및 비상용)
# - Q     → 종료

import os
import cv2
import time
import json
import queue
import torch
import threading
import requests
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 방지

# =========================
# 환경설정
# =========================
YOLO_WEIGHTS = "/home/eyedia/Downloads/EYEDIA_MODEL-feat-68-integration/scripts/gazedetection/best.pt"
CLIP_ID = "openai/clip-vit-base-patch32"
FAISS_INDEX = "faiss/met_text.index"
FAISS_META = "faiss/met_structured_with_objects.json"
MODEL_URL = "http://3.34.240.201:8000"

SCENE_CAM_PATH = "/dev/video0"
EYE_CAM_PATH = "/dev/video2"
HID_DEVICE = "/dev/input/by-id/usb-1d57_ad02-event-kbd"

REQUEST_TIMEOUT = 5.0
GAZE_MODEL_PATH = "scripts/gazedetection/3-best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 모델 로드
# =========================
print("🚀 Loading models...")
yolo_model = YOLO(YOLO_WEIGHTS)
clip_model = CLIPModel.from_pretrained(CLIP_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

# Gaze model (YOLOv8 4-class classification)
try:
    gaze_model = YOLO(GAZE_MODEL_PATH)
    print(f"✅ Gaze model loaded: {GAZE_MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load gaze model: {e}")
    gaze_model = None

# FAISS 로드
try:
    import faiss

    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_META, "r", encoding="utf-8") as f:
        image_meta = json.load(f)
except Exception as e:
    print(f"[WARN] FAISS load 실패: {e}")
    index, image_meta = None, []

# =========================
# 유틸 함수
# =========================
def embed_crop(image_bgr):
    """BGR 이미지를 CLIP 임베딩 벡터로 변환 (FAISS 검색용)"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = clip_processor(images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype("float32")


def choose_best_box(results):
    """YOLO 탐지 결과 중 가장 점수가 높은 박스를 선택"""
    best = None
    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w == 0 or h == 0:
            continue
        score = conf * np.sqrt(w * h)
        if best is None or score > best[-1]:
            best = (x1, y1, x2, y2, score)
    return best


def detect_art_id(frame_bgr):
    """YOLO 탐지 → FAISS 검색 → art_id 반환"""
    res = yolo_model(frame_bgr, verbose=False)[0]
    best = choose_best_box(res)
    if not best:
        return None
    x1, y1, x2, y2, score = best
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # FAISS 검색 
    if index:
        qv = embed_crop(crop).reshape(1, -1)
        D, I = index.search(qv, k=1)
        idx = int(I[0][0])
        art_id = str(image_meta[idx]["full_image_id"])
    else:
        art_id = "UNKNOWN"

    return art_id, (x1, y1, x2, y2), score


def draw_box(frame, box, color, text):
    """OpenCV 프레임에 사각형과 텍스트 표시"""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def get_gaze_q(frame_bgr, fallback_q="Q2"):
    """Gaze 모델로 4사분면 Q값 예측"""
    if gaze_model is None:
        print("[WARN] Gaze model not loaded, fallback.")
        return fallback_q
    try:
        results = gaze_model(frame_bgr, verbose=False)
        res = results[0]
        pred_idx = res.probs.top1
        class_name = res.names[pred_idx]    
        q = class_name.upper()
        if q in ("Q1", "Q2", "Q3", "Q4"):
            # 
            if q=="Q1":
                return "Q2"
            if q=="Q2":
                return "Q1"
            if q=="Q3":
                return "Q4"
            if q=="Q4":
                return "Q3"

            # return q
        print(f"[WARN] Gaze: 알 수 없는 클래스 이름: {class_name}")
        return fallback_q
    except Exception as e:
        print(f"[WARN] Gaze prediction failed: {e}")
        return fallback_q


def post_model_detect(art_id):
    """art_id만 서버로 전송 (/detect-art)"""
    try:
        r = requests.post(f"{MODEL_URL}/process-image?art_id={art_id}", timeout=REQUEST_TIMEOUT)
        print(f"🎯 detect-art sent: {r.status_code}")
    except Exception as e:
        print(f"[WARN] detect-art failed: {e}")


def post_model_detect_area(art_id, q):
    """art_id + Q값 서버 전송 (/detect-area)"""
    try:
        r = requests.post(f"{MODEL_URL}/process-image?art_id={art_id}&q={q}", timeout=REQUEST_TIMEOUT)
        print(f"🗺️ detect-area sent: {r.status_code}")
    except Exception as e:
        print(f"[WARN] detect-area failed: {e}")


def open_capture(dev, size=(640, 480), fps=30):
    """지정된 경로의 카메라 열기"""
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(10):
        cap.read()
    ok, _ = cap.read()
    if ok:
        return cap
    cap.release()
    return None


# =========================
# HID / 콘솔 입력
# =========================
class KeyEvent:
    TAB, ENTER, Q, NUM1, NUM2, NUM3, NUM4 = "TAB", "ENTER", "Q", "1", "2", "3", "4"


def hid_reader(dev_path, out_q):
    """리모컨(HID) 입력을 큐로 전달"""
    try:
        import evdev
        device = evdev.InputDevice(dev_path)
        print(f"🔌 HID 연결: {device.path} ({device.name})")
        for event in device.read_loop():
            if event.type != evdev.ecodes.EV_KEY:
                continue
            key_event = evdev.categorize(event)
            if key_event.keystate != evdev.KeyEvent.key_down:
                continue
            code = key_event.scancode
            if code == evdev.ecodes.KEY_TAB:
                out_q.put(KeyEvent.TAB)
            elif code in (evdev.ecodes.KEY_ENTER, evdev.ecodes.KEY_KPENTER):
                out_q.put(KeyEvent.ENTER)
            elif code == evdev.ecodes.KEY_1:
                out_q.put(KeyEvent.NUM1)
            elif code == evdev.ecodes.KEY_2:
                out_q.put(KeyEvent.NUM2)
            elif code == evdev.ecodes.KEY_3:
                out_q.put(KeyEvent.NUM3)
            elif code == evdev.ecodes.KEY_4:
                out_q.put(KeyEvent.NUM4)
            elif code == evdev.ecodes.KEY_Q:
                out_q.put(KeyEvent.Q)
                break
    except Exception:
        print("[INFO] HID 없음/권한 문제 → 콘솔 입력 폴백.")
        while True:
            key = input("키 [tab/enter/1/2/3/4/q]: ").strip().lower()
            if key == "tab":
                out_q.put(KeyEvent.TAB)
            elif key == "enter":
                out_q.put(KeyEvent.ENTER)
            elif key in ("1", "2", "3", "4"):
                out_q.put(key)
            elif key == "q":
                out_q.put(KeyEvent.Q)
                break


# =========================
# 메인 루프
# =========================
def main():
    print("✅ EYEDIA Dual-Cam Remote (Gaze Model)")

    scene = open_capture(SCENE_CAM_PATH, size=(1280, 720), fps=30)
    eye = open_capture(EYE_CAM_PATH, size=(320, 240), fps=15)
    if not scene or not eye:
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    key_q = queue.Queue()
    threading.Thread(target=hid_reader, args=(HID_DEVICE, key_q), daemon=True).start()

    selected_q = "Q2"
    overlay = "Ready"

    try:
        while True:
            ok_s, scene_frame = scene.read()
            ok_e, eye_frame = eye.read()
            if not ok_s or not ok_e:
                time.sleep(0.03)
                continue

            try:
                key = key_q.get_nowait()
            except queue.Empty:
                key = None

            # 수동 Q 변경
            if key in ("1", "2", "3", "4"):
                selected_q = f"Q{key}"
                print(f"[MANUAL] Q → {selected_q}")

            # 종료
            if key == KeyEvent.Q:
                print("👋 종료(Q).")
                break

            # TAB → detect-art
            if key == KeyEvent.TAB:
                print("▶ TAB: detect-art")
                out = detect_art_id(scene_frame)
                if out:
                    art_id, box, conf = out
                    draw_box(scene_frame, box, (255, 0, 0), f"art:{art_id}")
                    overlay = f"ART {art_id}"
                    post_model_detect(art_id)

            # ENTER → detect-area
            if key == KeyEvent.ENTER:
                print("▶ ENTER: detect-area (Gaze)")
                auto_q = get_gaze_q(eye_frame, fallback_q=selected_q)
                if auto_q != selected_q:
                    selected_q = auto_q
                    print(f"[GAZE] Q → {selected_q}")

                out = detect_art_id(scene_frame)
                if out:
                    art_id, box, conf = out
                    draw_box(scene_frame, box, (0, 255, 255), f"art:{art_id} {selected_q}")
                    overlay = f"AREA {art_id} {selected_q}"
                    post_model_detect_area(art_id, selected_q)

            # 화면 출력
            info = f"Q:{selected_q} | {overlay}"
            cv2.putText(scene_frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 255), 2)
            cv2.imshow("Scene (Artwork)", scene_frame)
            cv2.imshow("Eye (Tracking)", eye_frame)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        scene.release()
        eye.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
