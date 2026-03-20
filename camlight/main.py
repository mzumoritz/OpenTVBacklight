import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
import os
import threading
import requests
from flask import Flask, render_template, Response, request, jsonify

# ---------------------------------------------------------------------------
# Config — read from HA add-on options file
# ---------------------------------------------------------------------------
OPTIONS_FILE = "/data/options.json"

def load_options() -> dict:
    if os.path.exists(OPTIONS_FILE):
        with open(OPTIONS_FILE) as f:
            return json.load(f)
    return {}

options = load_options()

MQTT_HOST     = options.get("mqtt_host", "core-mosquitto")
MQTT_PORT     = int(options.get("mqtt_port", 1883))
MQTT_TOPIC    = options.get("mqtt_topic", "tv/backlight/color")
MQTT_USER     = options.get("mqtt_user", "")
MQTT_PASSWORD = options.get("mqtt_password", "")
INTERVAL      = int(options.get("capture_interval_ms", 200)) / 1000.0

# HA Supervisor API
SUPERVISOR_TOKEN = os.getenv("SUPERVISOR_TOKEN", "")
HA_API_BASE      = "http://supervisor/core/api"

DISPLAY_WIDTH  = 640
DISPLAY_HEIGHT = 480

ROI_FILE = "/data/roi.json"
CAM_FILE = "/data/camera.json"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)

# ---------------------------------------------------------------------------
# HA Supervisor API helpers
# ---------------------------------------------------------------------------
def ha_headers() -> dict:
    return {
        "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
        "Content-Type": "application/json"
    }

def get_ha_cameras() -> list:
    try:
        r = requests.get(f"{HA_API_BASE}/states", headers=ha_headers(), timeout=5)
        r.raise_for_status()
        states = r.json()
        cameras = [
            {
                "entity_id": s["entity_id"],
                "name": s["attributes"].get("friendly_name", s["entity_id"])
            }
            for s in states
            if s["entity_id"].startswith("camera.")
        ]
        return cameras
    except Exception as e:
        print(f"[ha] Failed to fetch camera list: {e}")
        return []

def fetch_ha_frame(entity_id: str):
    try:
        url = f"{HA_API_BASE}/camera_proxy/{entity_id}"
        r = requests.get(url, headers=ha_headers(), timeout=5)
        r.raise_for_status()
        img_array = np.frombuffer(r.content, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[ha] Failed to fetch frame from {entity_id}: {e}")
        return None

# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------
def crop_to_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = int(roi["x1"] * w)
    y1 = int(roi["y1"] * h)
    x2 = int(roi["x2"] * w)
    y2 = int(roi["y2"] * h)
    cropped = frame[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else frame

def draw_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    x1 = int(roi["x1"] * w)
    y1 = int(roi["y1"] * h)
    x2 = int(roi["x2"] * w)
    y2 = int(roi["y2"] * h)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out

# ---------------------------------------------------------------------------
# Thread 1: Capture
# ---------------------------------------------------------------------------
def capture_loop():
    global latest_frame

    usb_cap = None

    while True:
        cam_config = load_json(CAM_FILE)

        if not cam_config:
            time.sleep(1)
            continue

        mode      = cam_config.get("mode", "usb")
        entity_id = cam_config.get("entity_id", "")

        if mode == "usb":
            if usb_cap is None or not usb_cap.isOpened():
                print("[capture] Opening USB camera...")
                usb_cap = cv2.VideoCapture(0)
                usb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                usb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                usb_cap.set(cv2.CAP_PROP_AUTO_WB, 0)

            ret, frame = usb_cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame.copy()
            else:
                print("[capture] USB read failed, retrying...")
                usb_cap.release()
                usb_cap = None
                time.sleep(1)

        elif mode == "ha_camera":
            if usb_cap is not None:
                usb_cap.release()
                usb_cap = None

            frame = fetch_ha_frame(entity_id)
            if frame is not None:
                with frame_lock:
                    latest_frame = frame.copy()

            time.sleep(INTERVAL)

# ---------------------------------------------------------------------------
# Thread 2: Color
# ---------------------------------------------------------------------------
def color_loop():
    client = mqtt.Client()
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

    connected = False
    while not connected:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            connected = True
        except Exception as e:
            print(f"[color] MQTT connection failed: {e}, retrying in 5s...")
            time.sleep(5)

    client.loop_start()
    print("[color] MQTT connected")

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        roi = load_json(ROI_FILE)
        region = crop_to_roi(frame, roi) if roi else frame

        avg = region.mean(axis=(0, 1))  # BGR
        b, g, r = int(avg[0]), int(avg[1]), int(avg[2])

        payload = json.dumps({"r": r, "g": g, "b": b})
        client.publish(MQTT_TOPIC, payload)

        cam_config = load_json(CAM_FILE)
        if cam_config and cam_config.get("mode") == "usb":
            time.sleep(INTERVAL)
        else:
            time.sleep(0.05)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="/web/templates")

def generate_stream():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.05)
            continue

        roi = load_json(ROI_FILE)
        if roi:
            frame = draw_roi(frame, roi)

        display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        _, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def stream():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/cameras", methods=["GET"])
def list_cameras():
    cameras = get_ha_cameras()
    return jsonify(cameras)

@app.route("/camera", methods=["GET"])
def get_camera():
    return jsonify(load_json(CAM_FILE))

@app.route("/camera", methods=["POST"])
def set_camera():
    data = request.json
    save_json(CAM_FILE, data)
    print(f"[web] Camera source set: {data}")
    return jsonify({"status": "ok"})

@app.route("/roi", methods=["GET"])
def get_roi():
    return jsonify(load_json(ROI_FILE))

@app.route("/roi", methods=["POST"])
def save_roi():
    data = request.json
    save_json(ROI_FILE, data)
    print(f"[web] ROI saved: {data}")
    return jsonify({"status": "ok"})

@app.route("/roi/reset", methods=["POST"])
def reset_roi():
    if os.path.exists(ROI_FILE):
        os.remove(ROI_FILE)
    print("[web] ROI reset")
    return jsonify({"status": "ok"})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_capture = threading.Thread(target=capture_loop, daemon=True, name="capture")
    t_capture.start()

    time.sleep(1)

    t_color = threading.Thread(target=color_loop, daemon=True, name="color")
    t_color.start()

    print("[web] Starting on port 8099")
    app.run(host="0.0.0.0", port=8099, threaded=True)