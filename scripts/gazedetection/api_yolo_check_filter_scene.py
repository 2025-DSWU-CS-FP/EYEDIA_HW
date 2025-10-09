#!/usr/bin/env python3
# EYEDIA 스마트 아이웨어 컨트롤러 (모델 서버 2개 GET API 전용)
# - TAB   : GET {MODEL_URL}/process-image?art_id=<ART_ID_HINT>  → painting_id 저장
# - ENTER : GET {MODEL_URL}/process-image?art_id=<painting_id>&q=Q<zone>
#
# 모델 URL 기본값: http://3.34.240.201:8000
# 필요 ENV:
#   MODEL_URL    (옵션) 기본값 위와 동일
#   ART_ID_HINT  (옵션) 기본 200002 (200003도 테스트 가능)
#   EYE_CAM_PATH / REMOTE_EVENT (옵션)

import os, sys, time, signal, threading
import requests, cv2
from evdev import InputDevice, ecodes

# ---------- gaze 모듈 ----------
try:
    import gaze_detection as gaze
except ImportError:
    print("CRITICAL: 'gaze_detection.py'가 필요합니다. 동일 폴더에 두세요.")
    sys.exit(1)

# ---------- 설정 ----------
MODEL_URL   = os.environ.get("MODEL_URL", "http://3.34.240.201:8000").rstrip("/")
PROCESS_IMAGE_ENDPOINT = "/process-image"

ART_ID_HINT = int(os.environ.get("ART_ID_HINT", "200002"))

EYE_CAM_PATH = os.environ.get(
    "EYE_CAM_PATH",
    "/dev/v4l/by-id/usb-Generic_USB2.0_PC_CAMERA-video-index0"
)
REMOTE_EVENT_PATH = os.environ.get(
    "REMOTE_EVENT",
    "/dev/input/by-id/usb-1d57_ad02-event-kbd"
)

REQUEST_TIMEOUT = (5, 20)  # (connect, read)
DEBOUNCE_SEC = 0.45
CAPTURE_RETRIES = 3
WARMUP_FRAMES = 2

# ---------- 상태 ----------
running = True
busy_tab = threading.Event()
busy_enter = threading.Event()

_painting_id = None
_id_lock = threading.Lock()


def set_painting_id(pid: int):
    global _painting_id
    with _id_lock:
        _painting_id = pid

def get_painting_id():
    with _id_lock:
        return _painting_id

# ---------- 세션 & 로깅 ----------
session = requests.Session()
session.headers.update({"User-Agent": "EYEDIA-ModelOnly/1.0"})

_print_lock = threading.Lock()
def log(msg: str):
    with _print_lock:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

def dump_resp(tag, r):
    ct = r.headers.get("content-type", "")
    log(f"[{tag}] status={r.status_code}, content-type={ct}")
    try:
        if "application/json" in (ct or "").lower():
            log(f"[{tag}] json={r.json()}")
        else:
            log(f"[{tag}] text={r.text[:300]}")
    except Exception:
        log(f"[{tag}] text={r.text[:300]}")

# ---------- 유틸 ----------
# 상단 설정 근처(전역)에 추가: 헬스 실패 시 계속 진행할지
CONTINUE_ON_HEALTH_FAIL = True  # 필요하면 False로

def health_check():
    """모델 서버가 루트 404여도 실제 엔드포인트로 헬스 확인."""
    candidates = [
        ("/health", None),                         # 있으면 제일 깔끔
        ("/openapi.json", None),                   # FastAPI 기본 스키마
        ("/process-image", {"art_id": ART_ID_HINT})# 우리 실제 엔드포인트
    ]
    ok = False
    for path, params in candidates:
        url = f"{MODEL_URL.rstrip('/')}{path}"
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT[0], allow_redirects=False)
            dump_resp(f"HEALTH {path}", r)
            if 200 <= r.status_code < 400:
                ok = True
                break
        except requests.RequestException as e:
            log(f"HEALTH 예외({path}): {e}")

    if not ok:
        msg = "헬스체크 실패"       


        if CONTINUE_ON_HEALTH_FAIL:
            log(f"⚠️ {msg} — 계속 진행합니다(CONTINUE_ON_HEALTH_FAIL=True).")
            return True
        else:
            log(f"❌ {msg}")
            return False                                                                            
    return True


def open_camera(path: str):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def warmup_camera(cap: cv2.VideoCapture):
    for _ in range(WARMUP_FRAMES):
        cap.read()

def read_frame_with_retry(cap: cv2.VideoCapture, retries=CAPTURE_RETRIES):
    for i in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05 * (i + 1))
    return None

def q_normalize(zone) -> str:
    s = str(zone)
    if s.isdigit():
        return f"Q{int(s)}"
    return s if s.upper().startswith("Q") else f"Q{s}"

# ---------- TAB: painting_id 전송 (GET art_id=...) ----------
def task_send_artid():
    if busy_tab.is_set():
        log("⏳ TAB 작업 진행 중 → 무시")
        return
    busy_tab.set()
    try:
        url = f"{MODEL_URL}{PROCESS_IMAGE_ENDPOINT}"
        params = {"art_id": ART_ID_HINT}   # ← 쿼리스트링으로 전달
        log(f"➡️  MODEL POST {url} params={params}")
        r = session.post(url, params=params, timeout=REQUEST_TIMEOUT)
        dump_resp("MODEL_TAB", r)
        if not (200 <= r.status_code < 400):
            log("❌ MODEL_TAB 비정상 상태코드"); return

        # 응답에서 painting_id 저장(있으면)
        try:
            data = r.json()
        except Exception:
            log("⚠️ MODEL 응답 JSON 파싱 실패"); return
        pid = data.get("painting_id") or data.get("paintingId")
        if pid is not None:
            set_painting_id(int(pid))
            log(f"✅ painting_id 저장: {pid}")
        else:
            set_painting_id(ART_ID_HINT)
            log(f"⚠️ painting_id 미포함 → ART_ID_HINT({ART_ID_HINT}) 사용")
    except Exception as e:
        log(f"❌ TAB 예외: {e}")
    finally:
        busy_tab.clear()




# ---------- ENTER: painting_id + q 전송 (GET art_id=..., q=Qn) ----------
def task_send_area(eye_cap: cv2.VideoCapture):
    if busy_enter.is_set():
        log("⏳ ENTER 작업 진행 중 → 무시")
        return
    busy_enter.set()
    try:
        pid = get_painting_id() or ART_ID_HINT

        # 눈동자 프레임에서 zone 추정 (실패 시 Q2)
        frame = read_frame_with_retry(eye_cap)
        if frame is None:
            log("❌ 눈동자 프레임 캡처 실패 → Q2 기본값 사용")
            zone = 2
        else:
            zone = gaze.predict_zone(frame)

            
            if zone is None:
                log("⚠️ 시선 영역 예측 실패 → Q2 기본값 사용")
                zone = 2

        q = q_normalize(zone)  # "Q1"~"Q4"

        url = f"{MODEL_URL}{PROCESS_IMAGE_ENDPOINT}"
        params = {"art_id": int(pid), "q": q}   # ← 쿼리스트링으로 전달
        log(f"➡️  MODEL POST {url} params={params}")
        r = session.post(url, params=params, timeout=REQUEST_TIMEOUT)
        dump_resp("MODEL_ENTER", r)
        if 200 <= r.status_code < 400:
            log("✅ ENTER 요청 완료")
        else:
            log("❌ ENTER 비정상 상태코드")
    except Exception as e:
        log(f"❌ ENTER 예외: {e}")
    finally:
        busy_enter.clear()


# ---------- 메인 ----------
def main():
    log(f"MODEL_URL={MODEL_URL}")
    log(f"ART_ID_HINT={ART_ID_HINT}")

    if not health_check():
        log("헬스체크 실패 → 종료"); return

    eye_cap = open_camera(EYE_CAM_PATH)
    if not eye_cap.isOpened():
        log(f"치명적: 눈동자 카메라 열기 실패: {EYE_CAM_PATH}")
        return
    warmup_camera(eye_cap)

    try:
        dev = InputDevice(REMOTE_EVENT_PATH)
    except FileNotFoundError:
        eye_cap.release()
        log(f"치명적: 리모컨 장치 없음: {REMOTE_EVENT_PATH}")
        return

    log(f"리모컨 대기: {dev.path} ({dev.name})")
    log(" - TAB  : GET /process-image?art_id=<ART_ID_HINT>")
    log(" - ENTER: GET /process-image?art_id=<painting_id>&q=Q<zone>")
    log(" - Q/BACK/EXIT: 종료")

    def _shutdown_handler(sig, frm):
        global running
        if running:
            log("종료 신호 수신 → 안전 종료")
            running = False
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    last_press = 0.0
    threads = []

    try:
        for event in dev.read_loop():
            if not running:
                break
            if event.type != ecodes.EV_KEY or event.value != 1:  # key down만
                continue

            now = time.monotonic()
            if now - last_press < DEBOUNCE_SEC:
                continue
            last_press = now

            code = event.code
            if code == ecodes.KEY_TAB:
                t = threading.Thread(target=task_send_artid, daemon=True)
                t.start(); threads.append(t)
            elif code in (ecodes.KEY_ENTER, ecodes.KEY_KPENTER):
                t = threading.Thread(target=task_send_area, args=(eye_cap,), daemon=True)
                t.start(); threads.append(t)
            elif code in (ecodes.KEY_Q, ecodes.KEY_BACK, ecodes.KEY_EXIT):
                log("종료 키 입력"); break
    except Exception as e:
        log(f"이벤트 루프 오류: {e}")
    finally:
        log("스레드 종료 대기...")
        for t in threads:
            t.join(timeout=1.0)
        log("자원 해제...")
        eye_cap.release()
        log("종료.")

if __name__ == "__main__":
    main()
