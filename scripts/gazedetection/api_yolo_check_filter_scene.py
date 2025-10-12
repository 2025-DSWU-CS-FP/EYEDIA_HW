#!/usr/bin/env python3
# eyedia_dualcam_remote.pyV
# TAB   → detect-art    → POST http://3.34.240.201:8000/process-image?art_id=...
# ENTER → detect-area   → POST http://3.34.240.201:8000/process-image?art_id=...&q=Qn
# 1~4   → Q1~Q4 수동 전환, Q/ESC → 종료

import os
import cv2
import time
import json
import faiss
import torch
import queue
import threading
import requests
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 방지

# =========================
# 환경설정
# =========================
YOLO_WEIGHTS   = "best.pt"  # 우리가 학습한 가중치
CLIP_ID        = "openai/clip-vit-base-patch32"
FAISS_INDEX    = "./faiss/met_text.index"
FAISS_META     = "./faiss/met_structured_with_objects.json"

MODEL_URL      = "http://3.34.240.201:8000"         

# 카메라 경로
SCENE_CAM_PATH = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_E03BCAAF-video-index0"  # 작품 카메라
EYE_CAM_PATH   = "/dev/video2"  # 눈동자 카메라 (환경에 따라 수정 가능)
HID_DEVICE     = "/dev/input/by-id/usb-1d57_ad02-event-kbd"

REQUEST_TIMEOUT = 5.0

# =========================
# 모델/인덱스 로드
# =========================
print("🚀 Load models...")
yolo_model = YOLO(YOLO_WEIGHTS)
clip_model = CLIPModel.from_pretrained(CLIP_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

index = faiss.read_index(FAISS_INDEX)
with open(FAISS_META, "r", encoding="utf-8") as f:
    image_meta = json.load(f)

# =========================
# gaze 모듈
# =========================
try:
    import gaze_detection as gd
    HAS_GAZE = True
except Exception as e:
    print(f"[INFO] gaze_detection 불러오기 실패(폴백): {e}")
    HAS_GAZE = False

# =========================
# 유틸 함수
# =========================
def open_eye_cam():
    # 1) 먼저 V4L2 경로
    cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)  # 보수적 시작
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        for _ in range(15): cap.read()
        ok, _ = cap.read()
        if ok:
            print("🎥 EYE V4L2 OK (YUYV 640x480@15)")
            return cap
        cap.release()

    # 2) 실패 시 GStreamer 우회
    gst = ("v4l2src device=/dev/video2 io-mode=2 do-timestamp=true ! "
           "video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! "
           "videoconvert ! appsink drop=true max-buffers=1 sync=false")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        for _ in range(10): cap.read()
        ok, _ = cap.read()
        if ok:
            print("🎥 EYE GST OK (YUY2→appsink)")
            return cap

    return None

def open_capture_strict(dev, prefer="MJPG", size=(640, 480), fps=30):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    try_list = [prefer, "YUYV"] if prefer == "MJPG" else ["YUYV", "MJPG"]
    for fourcc in try_list:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(10):
            cap.read()
        ok, _ = cap.read()
        if ok:
            print(f"🎥 Opened {dev} with {fourcc} {size[0]}x{size[1]}@{fps}")
            return cap
    cap.release()
    return None

def embed_crop(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = clip_processor(images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype("float32")

def choose_best_box(results):
    best = None
    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0
        w, h = max(0, x2-x1), max(0, y2-y1)
        if w == 0 or h == 0:
            continue
        score = conf * np.sqrt(w * h)
        if best is None or score > best[-1]:
            best = (x1, y1, x2, y2, score)
    return best

def detect_art_id(frame_bgr):
    res = yolo_model(frame_bgr, verbose=False)[0]
    best = choose_best_box(res)
    if not best:
        return None
    x1, y1, x2, y2, score = best
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    qv = embed_crop(crop).reshape(1, -1)
    D, I = index.search(qv, k=1)
    idx = int(I[0][0])
    art_id = str(image_meta[idx]["full_image_id"])
    return art_id, (x1, y1, x2, y2), score, float(D[0][0])

def get_gaze_q(eye_frame_bgr, fallback_q="Q2"):
    if not HAS_GAZE:
        return fallback_q
    try:
        zone = gd.predict_zone(eye_frame_bgr)
        if zone in (1, 2, 3, 4):
            return f"Q{zone}"
    except Exception as e:
        print(f"[WARN] gaze 예측 실패: {e}")
    return fallback_q

def post_model_detect(art_id):
    url = f"{MODEL_URL}/process-image?art_id={art_id}"
    try:
        r = requests.post(url, timeout=REQUEST_TIMEOUT)
        print(f"🎯 detect-art 전송 완료: {r.status_code}")
        return r
    except requests.RequestException as e:
        print(f"[WARN] 요청 실패: {e}")
        return None

def post_model_detect_area(art_id, q):
    url = f"{MODEL_URL}/process-image?art_id={art_id}&q={q}"
    try:
        r = requests.post(url, timeout=REQUEST_TIMEOUT)
        print(f"🗺️ detect-area 전송 완료: {r.status_code}")
        return r
    except requests.RequestException as e:
        print(f"[WARN] 요청 실패: {e}")
        return None

def draw_box(frame, box, color, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# =========================
# HID 리더
# =========================
class KeyEvent:
    TAB, ENTER, Q, NUM1, NUM2, NUM3, NUM4 = "TAB", "ENTER", "Q", "1", "2", "3", "4"

def hid_reader(dev_path, out_q):
    try:
        import evdev
    except Exception:
        print("[INFO] evdev 미설치/권한 문제 → 콘솔 폴백.")
        while True:
            key = input("키 [tab/enter/1/2/3/4/q]: ").strip().lower()
            if key == "tab": out_q.put(KeyEvent.TAB)
            elif key == "enter": out_q.put(KeyEvent.ENTER)
            elif key in ("1", "2", "3", "4"): out_q.put(key)
            elif key == "q": out_q.put(KeyEvent.Q); break
        return

    try:
        device = evdev.InputDevice(dev_path)
        print(f"🔌 HID 연결: {device.path} ({device.name})")
        for event in device.read_loop():
            if event.type != evdev.ecodes.EV_KEY:
                continue
            key_event = evdev.categorize(event)
            if key_event.keystate != evdev.KeyEvent.key_down:
                continue
            code = key_event.scancode
            if code == evdev.ecodes.KEY_TAB: out_q.put(KeyEvent.TAB)
            elif code in (evdev.ecodes.KEY_ENTER, evdev.ecodes.KEY_KPENTER): out_q.put(KeyEvent.ENTER)
            elif code == evdev.ecodes.KEY_1: out_q.put(KeyEvent.NUM1)
            elif code == evdev.ecodes.KEY_2: out_q.put(KeyEvent.NUM2)
            elif code == evdev.ecodes.KEY_3: out_q.put(KeyEvent.NUM3)
            elif code == evdev.ecodes.KEY_4: out_q.put(KeyEvent.NUM4)
            elif code == evdev.ecodes.KEY_Q: out_q.put(KeyEvent.Q); break
    except FileNotFoundError:
        print(f"[WARN] HID 장치 없음: {dev_path} → 콘솔 폴백.")
        while True:
            key = input("키 [tab/enter/1/2/3/4/q]: ").strip().lower()
            if key == "tab": out_q.put(KeyEvent.TAB)
            elif key == "enter": out_q.put(KeyEvent.ENTER)
            elif key in ("1", "2", "3", "4"): out_q.put(key)
            elif key == "q": out_q.put(KeyEvent.Q); break

# =========================
# 메인
# =========================
def main():
    print("✅ EYEDIA Dual-Cam Remote")
    print(f"   YOLO: {YOLO_WEIGHTS}")
    print(f"   MODEL_URL: {MODEL_URL}")
    print(f"   SCENE_CAM: {SCENE_CAM_PATH}")
    print(f"   EYE_CAM  : {EYE_CAM_PATH}")
    print(f"   HID_DEVICE: {HID_DEVICE}")

    scene = open_capture_strict(SCENE_CAM_PATH, prefer="MJPG", size=(1280, 720), fps=30)
    eye = open_eye_cam()

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

            if key in ("1", "2", "3", "4"):
                selected_q = f"Q{key}"
                print(f"[MANUAL] Q → {selected_q}")
            if key == KeyEvent.Q:
                print("👋 종료(Q).")
                break

            if key == KeyEvent.TAB:
                print("▶ TAB: detect-art")
                out = detect_art_id(scene_frame)
                if out:
                    art_id, box, conf_like, dist = out
                    draw_box(scene_frame, box, (255, 0, 0), f"art:{art_id}")
                    overlay = f"ART {art_id}"
                    post_model_detect(art_id)

            if key == KeyEvent.ENTER:
                print("▶ ENTER: detect-area")
                auto_q = get_gaze_q(eye_frame, selected_q)
                if auto_q != selected_q:
                    selected_q = auto_q
                    print(f"[GAZE] Q → {selected_q}")
                print(f"[REQ]WILL SEND Q = {selected_q}")
                out = detect_art_id(scene_frame)
                if out:
                    art_id, box, conf_like, dist = out
                    draw_box(scene_frame, box, (0, 255, 255), f"art:{art_id} {selected_q}")
                    overlay = f"AREA {art_id} {selected_q}"
                    post_model_detect_area(art_id, selected_q)

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
