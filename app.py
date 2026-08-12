import os
import base64
import logging
import threading
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from tracker import ByteTracker

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
MODEL_XML      = os.path.join("models", "best_openvino_model", "best.xml")
CLASSES        = {0: "gun", 1: "intruder"}
CONF_THRESHOLD = 0.4
INPUT_SIZE     = (640, 640)
TRAIL_MAX_LEN  = 30   # max historical positions per track_id

# ──────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────
compiled_model = None
input_layer    = None
output_layer   = None
model_lock     = threading.Lock()

zones: list[list[dict]] = []   # [{x,y} normalized 0-1] per polygon
event_log: list[dict]  = []

# Per-track history: {track_id: [(cx_norm, cy_norm), ...]}
trail_history: dict[int, list[tuple]] = {}

tracker = ByteTracker(iou_threshold=0.35, max_age=10)
tracker_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────
def load_model():
    global compiled_model, input_layer, output_layer
    try:
        from openvino.runtime import Core
        ie = Core()
        model = ie.read_model(MODEL_XML)
        compiled_model = ie.compile_model(model, "CPU")
        input_layer    = compiled_model.input(0)
        output_layer   = compiled_model.output(0)
        log.info("✅ OpenVINO model loaded successfully.")
    except Exception as e:
        log.error(f"❌ Failed to load model: {e}")
        compiled_model = None

load_model()

# ──────────────────────────────────────────────────────────────
# Preprocessing / Postprocessing
# ──────────────────────────────────────────────────────────────
def preprocess(img_bgr):
    """Letterbox resize + normalize → (1,3,640,640) NCHW float32."""
    h, w   = img_bgr.shape[:2]
    scale  = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    padded  = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.uint8)
    top, left = (INPUT_SIZE[0] - nh) // 2, (INPUT_SIZE[1] - nw) // 2
    padded[top:top+nh, left:left+nw] = resized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC → 1CHW
    return blob, scale, top, left


def postprocess(output, scale, pad_top, pad_left, orig_h, orig_w, conf_thresh):
    """Parse YOLOv8 output (1, 6, 8400) → list of raw detection dicts."""
    pred = output[0].T  # (8400, 6) where 6 = cx,cy,w,h + n_classes scores

    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    class_scores   = pred[:, 4:]
    class_ids      = np.argmax(class_scores, axis=1)
    confidences    = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confidences >= conf_thresh
    raw  = []
    for i in np.where(mask)[0]:
        x1 = np.clip((cx[i] - bw[i] / 2 - pad_left) / scale, 0, orig_w)
        y1 = np.clip((cy[i] - bh[i] / 2 - pad_top)  / scale, 0, orig_h)
        x2 = np.clip((cx[i] + bw[i] / 2 - pad_left) / scale, 0, orig_w)
        y2 = np.clip((cy[i] + bh[i] / 2 - pad_top)  / scale, 0, orig_h)
        raw.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "confidence": float(confidences[i]),
            "class_id":   int(class_ids[i]),
            "class_name": CLASSES.get(int(class_ids[i]), "unknown"),
        })

    if not raw:
        return []

    # NMS
    boxes  = [[d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]] for d in raw]
    scores = [d["confidence"] for d in raw]
    idxs   = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.45)
    if isinstance(idxs, tuple) or len(idxs) == 0:
        return []
    return [raw[i] for i in idxs.flatten()]


def point_in_polygon(px, py, polygon_norm, img_w, img_h):
    if len(polygon_norm) < 3:
        return False
    pts = np.array([[p["x"] * img_w, p["y"] * img_h] for p in polygon_norm], dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0


# ──────────────────────────────────────────────────────────────
# Trail management
# ──────────────────────────────────────────────────────────────
def update_trail(track_id: int, cx_norm: float, cy_norm: float):
    """Append normalized center to trail history, keep last TRAIL_MAX_LEN."""
    if track_id not in trail_history:
        trail_history[track_id] = []
    trail_history[track_id].append((cx_norm, cy_norm))
    if len(trail_history[track_id]) > TRAIL_MAX_LEN:
        trail_history[track_id].pop(0)


def prune_trails(active_ids: set):
    """Remove trail data for tracks that have been gone too long."""
    stale = [tid for tid in list(trail_history) if tid not in active_ids]
    for tid in stale:
        # Keep trail for a while so fade-out works on frontend
        pass   # frontend will simply not receive it if track disappears


# ──────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ── POST /detect ─────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    if compiled_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing frame"}), 400

    conf_thresh = float(data.get("conf_threshold", CONF_THRESHOLD))

    # ── Decode frame ─────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(data["frame"].split(",")[-1])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Invalid image"}), 400
    except Exception as e:
        return jsonify({"error": f"Decode error: {e}"}), 400

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Inference ────────────────────────────────────────────
    with model_lock:
        blob, scale, pad_top, pad_left = preprocess(img_bgr)
        try:
            raw_output = compiled_model({input_layer: blob})[output_layer]
        except Exception as e:
            return jsonify({"error": f"Inference error: {e}"}), 500

    detections = postprocess(raw_output, scale, pad_top, pad_left, orig_h, orig_w, conf_thresh)

    # ── Tracking ─────────────────────────────────────────────
    with tracker_lock:
        detections = tracker.update(detections)

    # ── Zone check + trail update ─────────────────────────────
    current_zones      = zones.copy()
    intrusion_detected = False
    results            = []
    active_ids         = set()

    for det in detections:
        tid    = det.get("track_id", 0)
        foot_x = (det["x1"] + det["x2"]) / 2
        foot_y = det["y2"]
        cx     = (det["x1"] + det["x2"]) / 2
        cy     = (det["y1"] + det["y2"]) / 2

        in_zone = any(
            point_in_polygon(foot_x, foot_y, zone, orig_w, orig_h)
            for zone in current_zones
        )

        if in_zone and det["class_name"] == "intruder":
            intrusion_detected = True

        # Normalized coords for frontend
        det["x1n"]    = det["x1"] / orig_w
        det["y1n"]    = det["y1"] / orig_h
        det["x2n"]    = det["x2"] / orig_w
        det["y2n"]    = det["y2"] / orig_h
        det["in_zone"] = in_zone

        # Trail update
        update_trail(tid, cx / orig_w, cy / orig_h)
        active_ids.add(tid)
        det["trail"] = trail_history.get(tid, [])

        results.append(det)

    # Optionally log event
    if intrusion_detected and data.get("save_event", False):
        event_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "thumbnail": data["frame"],
            "track_ids": [d["track_id"] for d in results if d.get("in_zone") and d["class_name"] == "intruder"],
            "count":     sum(1 for d in results if d.get("in_zone") and d["class_name"] == "intruder"),
        })
        if len(event_log) > 100:
            event_log.pop(0)

    return jsonify({
        "detections":         results,
        "intrusion_detected": intrusion_detected,
        "zones_active":       len(current_zones),
        "frame_size":         {"w": orig_w, "h": orig_h},
    })


# ── Zone endpoints ───────────────────────────────────────────
@app.route("/zone", methods=["POST"])
def set_zone():
    data = request.get_json(force=True)
    if not data or "polygon" not in data:
        return jsonify({"error": "Missing polygon"}), 400
    zones.clear()
    if data["polygon"]:
        zones.append(data["polygon"])
    log.info(f"Zone updated: {len(zones[0]) if zones else 0} points")
    return jsonify({"status": "ok", "zones": len(zones)})


@app.route("/zone", methods=["GET"])
def get_zones():
    return jsonify({"zones": zones})


@app.route("/zone/clear", methods=["POST"])
def clear_zones():
    zones.clear()
    return jsonify({"status": "cleared"})


# ── Tracker reset ────────────────────────────────────────────
@app.route("/tracker/reset", methods=["POST"])
def reset_tracker():
    with tracker_lock:
        tracker.reset()
    trail_history.clear()
    return jsonify({"status": "tracker reset"})


# ── Events ──────────────────────────────────────────────────
@app.route("/events", methods=["GET"])
def get_events():
    return jsonify({"events": event_log[-50:]})


@app.route("/events/clear", methods=["POST"])
def clear_events():
    event_log.clear()
    return jsonify({"status": "cleared"})


# ── Status ──────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded":  compiled_model is not None,
        "zones_count":   len(zones),
        "events_count":  len(event_log),
        "tracks_active": len(tracker.tracks),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
