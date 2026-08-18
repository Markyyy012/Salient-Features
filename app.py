import base64
import os
import secrets
import tempfile
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from saliency_utils import (compute_static, draw_rois, extract_rois,
                            fine_threshold)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "objectness_trained_model")
UPLOAD_DIR = tempfile.mkdtemp(prefix="saliency_uploads_")
UPLOAD_FILES = {}

MOTION_WIDTH = 500
MOTION_HEIGHT = 375

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128),
]

app = Flask(__name__, static_folder="static", static_url_path="/static")


def to_png(img):
    if img.ndim == 2:
        bgr = img
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()


def load_image():
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        name = f.filename
    elif request.form.get("image"):
        name = os.path.basename(request.form["image"])
        img = cv2.imread(os.path.join(IMAGES_DIR, name))
    else:
        return None, None
    if img is None:
        return None, name
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), name


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/images")
def list_images():
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names = sorted(f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith(exts))
    return jsonify({"images": names})


@app.route("/api/static", methods=["POST"])
def static_saliency():
    image, name = load_image()
    if image is None:
        return jsonify({"error": f"Could not read image: {name}"}), 400

    spectral_map, fine_map = compute_static(image)
    thresh_map = fine_threshold(fine_map)
    min_area = 0.005 * image.shape[0] * image.shape[1]
    rois = extract_rois(thresh_map, min_area=min_area)
    roi_img = draw_rois(image, rois)

    return jsonify({
        "name": name,
        "size": [image.shape[0], image.shape[1]],
        "count": len(rois),
        "rois": [list(r) for r in rois],
        "images": {
            "original": to_png(image),
            "spectral": to_png(spectral_map),
            "fine": to_png(fine_map),
            "threshold": to_png(thresh_map),
            "roi": to_png(roi_img),
        },
    })


@app.route("/api/objectness", methods=["POST"])
def objectness_saliency():
    image, name = load_image()
    if image is None:
        return jsonify({"error": f"Could not read image: {name}"}), 400

    max_det = request.form.get("max_detections", type=int, default=10)
    max_det = max(1, min(max_det, 50))

    saliency = cv2.saliency.ObjectnessBING_create()
    saliency.setTrainingPath(MODEL_DIR)
    success, saliency_map = saliency.computeSaliency(image)
    if not success:
        return jsonify({"error": "Objectness computation failed"}), 500

    num = saliency_map.shape[0] if len(saliency_map.shape) > 1 else 1
    num = min(num, max_det)

    output = image.copy()
    proposals = []
    for i in range(num):
        det = saliency_map[i].flatten() if num > 1 else saliency_map.flatten()
        x1, y1, x2, y2 = [int(v) for v in det]
        color = PALETTE[i % len(PALETTE)]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        proposals.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return jsonify({
        "name": name,
        "size": [image.shape[0], image.shape[1]],
        "count": num,
        "proposals": proposals,
        "image": to_png(output),
    })


def make_synthetic_frame(i, tot=150, width=MOTION_WIDTH, height=MOTION_HEIGHT):
    frame = np.full((height, width, 3), 40, dtype="uint8")
    for idx, phase in enumerate((0, 90, 180)):
        cx = int(width * 0.25 + width * 0.5 * ((i + phase) % tot) / tot)
        cy = height // 2 + idx * 30
        cv2.circle(frame, (cx, cy), 25, (20, 180, 230), -1)
    return frame


def composite_side_by_side(frame, saliency_map):
    if saliency_map is None:
        saliency_map = np.zeros_like(frame)
    smap_bgr = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2BGR)
    combo = np.hstack([frame, smap_bgr])
    cv2.putText(combo, "Frame", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(combo, "Motion Saliency", (frame.shape[1] + 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return combo


def jpeg_part(buf):
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"


@app.route("/api/motion/synthetic")
def motion_synthetic():
    def gen():
        saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        saliency.setImagesize(MOTION_WIDTH, MOTION_HEIGHT)
        saliency.init()
        i = 0
        try:
            while True:
                frame = make_synthetic_frame(i)
                i += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, smap = saliency.computeSaliency(gray)
                if ok:
                    smap = (smap * 255).astype("uint8")
                else:
                    smap = None
                combo = composite_side_by_side(frame, smap)
                ok, buf = cv2.imencode(".jpg", combo,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                yield jpeg_part(buf.tobytes())
                time.sleep(0.05)
        except GeneratorExit:
            return
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/motion/upload", methods=["POST"])
def motion_upload():
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify({"error": "No video file"}), 400
    vid = secrets.token_hex(8)
    ext = os.path.splitext(f.filename)[1] or ".mp4"
    path = os.path.join(UPLOAD_DIR, vid + ext)
    f.save(path)
    UPLOAD_FILES[vid] = path
    return jsonify({"id": vid, "name": f.filename})


@app.route("/api/motion/stream/<vid>")
def motion_stream(vid):
    path = UPLOAD_FILES.get(vid)
    if path is None:
        return jsonify({"error": "Unknown video id"}), 404
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 404

    def gen():
        saliency = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(
                    frame, (MOTION_WIDTH,
                            int(frame.shape[0] * MOTION_WIDTH / frame.shape[1])))
                if saliency is None:
                    saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
                    saliency.setImagesize(frame.shape[1], frame.shape[0])
                    saliency.init()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, smap = saliency.computeSaliency(gray)
                if ok:
                    smap = (smap * 255).astype("uint8")
                else:
                    smap = None
                combo = composite_side_by_side(frame, smap)
                ok, buf = cv2.imencode(".jpg", combo,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                yield jpeg_part(buf.tobytes())
                time.sleep(0.05)
        except GeneratorExit:
            return
        finally:
            cap.release()
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


motion_sessions = {}
motion_lock = threading.Lock()


@app.route("/api/motion/frame", methods=["POST"])
def motion_frame():
    if "frame" not in request.files:
        return jsonify({"error": "No frame"}), 400
    data = request.files["frame"].read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Bad frame"}), 400

    width = MOTION_WIDTH
    height = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with motion_lock:
        sid = request.form.get("session_id", "")
        saliency = motion_sessions.get(sid)
        if saliency is None:
            saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            saliency.setImagesize(width, height)
            saliency.init()
            sid = secrets.token_hex(8)
            motion_sessions[sid] = saliency
        ok, smap = saliency.computeSaliency(gray)

    if ok:
        smap = (smap * 255).astype("uint8")
    else:
        smap = np.zeros_like(gray)
    ok, buf = cv2.imencode(".jpg", smap, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"X-Session-Id": sid})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
