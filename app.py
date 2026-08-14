import os
import base64
import logging
import threading
import time
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from tracker import ByteTracker
from model_loader import load_model, ModelInfo

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
CLASSES        = {0: "gun", 1: "intruder"}
CONF_THRESHOLD = 0.4
INPUT_SIZE     = (640, 640)
TRAIL_MAX_LEN  = 30   # max historical positions per track_id

# ──────────────────────────────────────────────────────────────
# CCTV Recording
# ──────────────────────────────────────────────────────────────

RECORDINGS_DIR = os.path.join(app.root_path, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

recording_writer = None
recording_filename = None
recording_last_intrusion = 0
recording_lock = threading.Lock()

RECORDING_POST_DELAY = 5
RECORDING_FPS = 5

# ──────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────
model_info: ModelInfo = None   # set at startup
model_lock  = threading.Lock()

zones: list[list[dict]] = []   # [{x,y} normalized 0-1] per polygon
event_log: list[dict]   = []

# Per-track history: {track_id: [(cx_norm, cy_norm), ...]}
trail_history: dict[int, list[tuple]] = {}

tracker      = ByteTracker(iou_threshold=0.35, max_age=10)
tracker_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────
# Model loading (startup, once)
# ──────────────────────────────────────────────────────────────
model_info = load_model()

def draw_recording_overlay(frame, detections, current_zones):
    """Draw detection boxes, labels, and zones onto the recorded CCTV frame."""
    output = frame.copy()
    h, w = output.shape[:2]

    # Draw zones
    for zone in current_zones:
        if len(zone) >= 3:
            pts = np.array(
                [[int(p["x"] * w), int(p["y"] * h)] for p in zone],
                dtype=np.int32
            )
            cv2.polylines(
                output,
                [pts],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2
            )

    # Draw detections
    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        in_zone = det.get("in_zone", False)
        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0.0)
        track_id = det.get("track_id", 0)

        if class_name == "intruder":
            box_color = (0, 0, 255) if in_zone else (0, 255, 0)
        else:
            box_color = (255, 0, 0)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        label = f"{class_name} ID:{track_id} {confidence:.2f}"

        if in_zone:
            label += " IN ZONE"

        text_y = max(y1 - 10, 20)

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA
        )

    return output


def start_recording(frame):
    global recording_writer
    global recording_filename

    with recording_lock:
        if recording_writer is not None:
            return

        height, width = frame.shape[:2]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intrusion_{timestamp}.mp4"
        filepath = os.path.join(RECORDINGS_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            filepath,
            fourcc,
            RECORDING_FPS,
            (width, height)
        )

        if not writer.isOpened():
            log.error("Gagal membuka VideoWriter: %s", filepath)
            return

        recording_writer = writer
        recording_filename = filename

        log.info("CCTV recording dimulai: %s", filename)
        return filename


def write_recording_frame(frame):
    global recording_writer

    with recording_lock:
        if recording_writer is not None:
            recording_writer.write(frame)


def stop_recording():
    global recording_writer
    global recording_filename

    with recording_lock:
        if recording_writer is None:
            return

        filename = recording_filename

        recording_writer.release()

        recording_writer = None
        recording_filename = None

        log.info("CCTV recording selesai: %s", filename)
        return filename


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


def postprocess_pytorch(results, orig_h, orig_w, conf_thresh):
    """Parse Ultralytics YOLO result objects → list of detection dicts."""
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "x1": float(np.clip(x1, 0, orig_w)),
                "y1": float(np.clip(y1, 0, orig_h)),
                "x2": float(np.clip(x2, 0, orig_w)),
                "y2": float(np.clip(y2, 0, orig_h)),
                "confidence": conf,
                "class_id":   cls_id,
                "class_name": CLASSES.get(cls_id, "unknown"),
            })
    return detections


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
# Inference dispatcher (works for both backends)
# ──────────────────────────────────────────────────────────────
def run_inference_on_frame(img_bgr, conf_thresh):
    """
    Dispatch inference to the correct backend.
    Returns list of detection dicts (same format for both backends).
    Raises RuntimeError if model is not loaded.
    """
    if not model_info.is_loaded:
        raise RuntimeError(
            "Model tidak dapat dimuat. "
            "Pastikan file best.pt atau best_openvino_model tersedia di folder project."
        )

    orig_h, orig_w = img_bgr.shape[:2]

    if model_info.backend == "openvino":
        blob, scale, pad_top, pad_left = preprocess(img_bgr)
        raw_output = model_info.compiled_model(
            {model_info.input_layer: blob}
        )[model_info.output_layer]
        return postprocess(raw_output, scale, pad_top, pad_left, orig_h, orig_w, conf_thresh)

    elif model_info.backend == "pytorch":
        results = model_info.pt_model.predict(
            source=img_bgr,
            conf=conf_thresh,
            verbose=False,
            device=model_info.device,
        )
        return postprocess_pytorch(results, orig_h, orig_w, conf_thresh)

    else:
        raise RuntimeError("Backend tidak dikenal: " + model_info.backend)


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
    if not model_info.is_loaded:
        return jsonify({
            "error": (
                "Model tidak dapat dimuat. "
                "Pastikan file best.pt atau best_openvino_model tersedia di folder project."
            )
        }), 503

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
        try:
            detections = run_inference_on_frame(img_bgr, conf_thresh)
        except Exception as e:
            return jsonify({"error": f"Inference error: {e}"}), 500

    # ── Tracking (ByteTracker — same for both backends) ──────
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
        foot_y = (det["y1"] + det["y2"]) / 2
        cx     = (det["x1"] + det["x2"]) / 2
        cy     = (det["y1"] + det["y2"]) / 2

        in_zone = any(
            point_in_polygon(
                foot_x,
                foot_y,
                zone,
                orig_w,
                orig_h
            )
            for zone in current_zones
        )

        # Deteksi objek di dalam zona dianggap sebagai intrusi
        if in_zone:
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

    # ──────────────────────────────────────────────────────────
    # CCTV Recording
    # ──────────────────────────────────────────────────────────

    global recording_last_intrusion

    current_time = time.time()

    log.info(
        "ZONE CHECK | detections=%d | zones=%d | intrusion=%s",
        len(results),
        len(current_zones),
        intrusion_detected
    )
    
    # Draw detection results onto the frame before saving it.
    recorded_frame = draw_recording_overlay(
        img_bgr,
        results,
        current_zones
    )

    log.info(
        "DETECTION: intrusion=%s | detections=%s",
        intrusion_detected,
        [
            {
                "class": d.get("class_name"),
                "track_id": d.get("track_id"),
                "in_zone": d.get("in_zone")
            }
            for d in results
        ]
    )

    
    if intrusion_detected:
        recording_last_intrusion = current_time

        if recording_writer is None:
            start_recording(recorded_frame)

        write_recording_frame(recorded_frame)

    elif recording_writer is not None:
        # Keep recording a few seconds after the last intrusion.
        write_recording_frame(recorded_frame)

        if current_time - recording_last_intrusion >= RECORDING_POST_DELAY:
            stop_recording()

    # The filename is returned to the frontend so the event log
    # can display the corresponding CCTV recording.
    active_recording_filename = recording_filename

    # Optionally log event
    if intrusion_detected and data.get("save_event", False):
        event_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "thumbnail": data["frame"],
            "track_ids": [d["track_id"] for d in results if d.get("in_zone") and d["class_name"] == "intruder"],
            "count":     sum(1 for d in results if d.get("in_zone") and d["class_name"] == "intruder"),
            "recording":  (
                f"/recordings/{active_recording_filename}"
                if active_recording_filename else None
            ),
            "recording_filename": active_recording_filename,
        })
        if len(event_log) > 100:
            event_log.pop(0)

    return jsonify({
        "detections":         results,
        "intrusion_detected": intrusion_detected,
        "zones_active":       len(current_zones),
        "frame_size":         {"w": orig_w, "h": orig_h},
        "recording":           (
            f"/recordings/{active_recording_filename}"
            if active_recording_filename else None
        ),
        "recording_filename":  active_recording_filename,
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

@app.route("/recordings", methods=["GET"])
def list_recordings():
    files = []

    if os.path.exists(RECORDINGS_DIR):
        for filename in os.listdir(RECORDINGS_DIR):
            if filename.lower().endswith(".mp4"):
                filepath = os.path.join(RECORDINGS_DIR, filename)

                files.append({
                    "filename": filename,
                    "url": f"/recordings/{filename}",
                    "created": datetime.fromtimestamp(
                        os.path.getctime(filepath)
                    ).isoformat(timespec="seconds")
                })

    files.sort(
        key=lambda x: x["created"],
        reverse=True
    )

    return jsonify({
        "recordings": files
    })

@app.route("/recordings/<path:filename>")
def get_recording(filename):
    return send_from_directory(
        RECORDINGS_DIR,
        filename,
        as_attachment=False
    )

@app.route("/recording/stop", methods=["POST"])
def manual_stop_recording():
    filename = stop_recording()
    return jsonify({
        "status": "stopped",
        "filename": filename,
        "url": f"/recordings/{filename}" if filename else None
    })


@app.route("/events/clear", methods=["POST"])
def clear_events():
    event_log.clear()
    return jsonify({"status": "cleared"})


# ── Status ──────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded":        model_info.is_loaded,
        "backend":             model_info.backend,
        "device":              model_info.device,
        "quantization":        model_info.quantization,   # "int8" | "fp32" | "none"
        "cpu_name":            model_info.cpu_name,
        "cpu_brand":           model_info.cpu_brand,
        "ov_devices":          model_info.available_ov_devices,
        "hardware_label":      model_info.display_label,
        "zones_count":         len(zones),
        "events_count":        len(event_log),
        "tracks_active":       len(tracker.tracks),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)