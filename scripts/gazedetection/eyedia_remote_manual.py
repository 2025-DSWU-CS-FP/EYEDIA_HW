#!/usr/bin/env python3
# EYEDIA 리모컨 매뉴얼 모드 (POST with Query Params 버전)
# ─────────────────────────────────────────────────────────────────
# 키 매핑
# - TAB    : art_id=200002 선택 (POST /process-image?art_id=200002)
# - UP     : art_id=200003 선택 (POST /process-image?art_id=200003)
# - WIN    : Q1 전송 (POST /process-image?art_id=<선택됨>&q=Q1)
# - LEFT   : Q2 전송 (POST /process-image?art_id=<선택됨>&q=Q2)
# - ENTER  : Q3 전송 (POST /process-image?art_id=<선택됨>&q=Q3)
# - RIGHT  : Q4 전송 (POST /process-image?art_id=<선택됨>&q=Q4)
# - ESC    : 종료
#
# 환경변수(선택):
#   MODEL_URL     : 기본 http://3.34.240.201:8000
#   REMOTE_EVENT  : 기본 /dev/input/by-id/usb-1d57_ad02-event-kbd
#
# 중요: 모델 서버는 FastAPI 스타일로 art_id/q를 "query params"로 받음.
# 예) POST /process-image?art_id=200002&q=Q1

import os
import time
import signal
import requests
from evdev import InputDevice, ecodes

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

# ── 액션 ─────────────────────────────────────────────────────────
def select_art(art_id: int):
    """작품 선택(POST) & 현재 선택 상태 저장"""
    global current_art_id
    r = post_process_image({"art_id": int(art_id)}, tag=f"ART({art_id})")
    # 서버 성공/실패와 상관없이 '현장 수동 모드'에선 선택 상태를 기억해 사용
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
