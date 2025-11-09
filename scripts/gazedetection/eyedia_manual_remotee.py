import os
import time
import signal
import requests
from evdev import InputDevice, ecodes
import cv2

# ── 설정 ─────────────────────────────────────────────────────────
MODEL_URL = os.environ.get("MODEL_URL", "http://3.34.240.201:8000").rstrip("/")
PROCESS_IMAGE_ENDPOINT = "/process-image"
REMOTE_EVENT_PATH = os.environ.get(
    "REMOTE_EVENT",
    "/dev/input/by-id/usb-1d57_ad02-event-kbd"
)

REQUEST_TIMEOUT = (5, 15)   # (connect, read)
DEBOUNCE_SEC = 0.35

# ── 상태 ─────────────────────────────────────────────────────────
running = True
current_art_id = None
session = requests.Session()
session.headers.update({"User-Agent": "EYEDIA-RemoteManual/1.2"})

# ── 유틸 ─────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def dump_resp(tag, r):
    try:
        log(f"[{tag}] status={r.status_code}, json={r.json()}")
    except Exception:
        log(f"[{tag}] status={r.status_code}, text={r.text[:200]}")

def post_process_image(params: dict, tag: str):
    """POST /process-image 를 '쿼리 파라미터'로 호출"""
    url = f"{MODEL_URL}{PROCESS_IMAGE_ENDPOINT}"
    log(f"➡️  POST {url} params={params}")
    try:
        r = session.post(url, params=params, timeout=REQUEST_TIMEOUT)
        dump_resp(tag, r)
        return r
    except Exception as e:
        log(f"❌ {tag} 예외: {e}")
        return None

# ── 카메라 관련 함수 ───────────────────────────────────────────
def open_camera_for_3_seconds(cam_path):
    """카메라를 3초 동안 열기"""
    cap = cv2.VideoCapture(cam_path)
    
    # 카메라가 열리면 3초 동안 스트리밍을 유지
    if cap.isOpened():
        log(f"[INFO] {cam_path} 열기 성공. 3초 동안 켬.")
        start_time = time.time()
        while time.time() - start_time < 3:  # 3초 동안
            ret, frame = cap.read()
            if not ret:
                break
            # 실시간 영상 출력 (디버그용)
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        log(f"[ERROR] {cam_path} 열기 실패")

# ── 액션 ─────────────────────────────────────────────────────────
def select_art(art_id: int):
    """작품 선택(POST) & 현재 선택 상태 저장"""
    global current_art_id
    r = post_process_image({"art_id": int(art_id)}, tag=f"ART({art_id})")
    current_art_id = int(art_id)
    log(f"🎨 작품 선택 완료 → art_id={current_art_id}")

def send_quadrant(q: str):
    """선택된 작품이 있어야 Q 전송 가능"""
    if current_art_id is None:
        log("⚠️ 먼저 작품을 선택하세요 (TAB=200001, UP=200002). Q 전송 취소.")
        return
    post_process_image({"art_id": current_art_id, "q": q}, tag=f"Q({q})")

# ── 키 처리 ───────────────────────────────────────────────────────
def handle_key_event(code: int):
    # 작품 선택
    if code == ecodes.KEY_TAB:        # 15
        select_art(200001)
    elif code == ecodes.KEY_UP:       # 103
        select_art(200002)

    # Q1~Q4 전송
    elif code in (ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA):  # 125 / 126 (Windows 키)
        send_quadrant("Q1")
    elif code == ecodes.KEY_LEFT:     # 105
        send_quadrant("Q2")
    elif code in (ecodes.KEY_ENTER, ecodes.KEY_KPENTER):       # 28 / 96
        send_quadrant("Q3")
    elif code == ecodes.KEY_RIGHT:    # 106
        send_quadrant("Q4")

    # 종료
    elif code in (ecodes.KEY_ESC, ecodes.KEY_BACK, ecodes.KEY_EXIT):  # 1 / 158 / 174
        global running
        log("⏹ 종료키 입력 → 프로그램 종료")
        running = False

    # 1번과 2번 키 눌렀을 때 카메라를 3초 동안 켜기
    elif code == ecodes.KEY_1:  # 1번 키
        open_camera_for_3_seconds("/dev/video0")  # SCENE_CAM_PATH
        log("▶ TAB / 1번 키 → 전송")
        post_process_image({"art_id": current_art_id, "q": "Q1"}, tag="Q1 전송")

    elif code == ecodes.KEY_2:  # 2번 키
        open_camera_for_3_seconds("/dev/video1")  # EYE_CAM_PATH
        log("▶ 2번 키 → 전송")
        post_process_image({"art_id": current_art_id, "q": "Q2"}, tag="Q2 전송")

# ── 메인 루프 ─────────────────────────────────────────────────────
def main():
    log(f"MODEL_URL={MODEL_URL}")
    log(f"REMOTE_EVENT_PATH={REMOTE_EVENT_PATH}")
    log("키맵:")
    log("  TAB(15)  → art_id=200001 선택")
    log("  UP(103)  → art_id=200002 선택")
    log("  WIN(125/126) → Q1,  LEFT(105) → Q2,  ENTER(28/96) → Q3,  RIGHT(106) → Q4")
    log("  ESC(1)/BACK(158)/EXIT(174) → 종료")
    log("※ 반드시 '작품 선택' 후 Q1~Q4를 전송하세요.")

    # 리모컨 장치 열기
    try:
        dev = InputDevice(REMOTE_EVENT_PATH)
    except FileNotFoundError:
        log(f"❌ 리모컨 장치 없음: {REMOTE_EVENT_PATH}")
        return

    log(f"🎮 리모컨 연결됨: {dev.path} ({dev.name})")

    last_press = 0.0

    def _shutdown_handler(sig, frm):
        global running
        if running:
            log("신호 수신 → 안전 종료")
            running = False
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        for event in dev.read_loop():
            if not running:
                break
            # key down만 처리 (event.value==1)
            if event.type != ecodes.EV_KEY or event.value != 1:
                continue

            now = time.monotonic()  
            if now - last_press < DEBOUNCE_SEC:
                continue
            last_press = now

            handle_key_event(event.code)
    except KeyboardInterrupt:
        log("🛑 수동 종료")
    except Exception as e:
        log(f"이벤트 루프 예외: {e}")
    finally:
        log("💤 종료 중...")
        try:
            dev.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
