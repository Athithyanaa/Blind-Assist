import os
import sys
import io
import base64
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import cv2
from transformers import DPTForDepthEstimation, DPTImageProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

yolo_model = None
depth_model = None
depth_processor = None


# ------------------------------------------------
# Load YOLOv8
# ------------------------------------------------
def get_yolo():
    global yolo_model
    if yolo_model is None:
        print("🔄 Loading YOLOv8n...")
        yolo_model = YOLO("yolov8n.pt")
        print("✅ YOLO loaded!")
    return yolo_model


# ------------------------------------------------
# Load Depth Model (SAFE working version)
# ------------------------------------------------
def get_depth_model():
    global depth_model, depth_processor

    if depth_model is None:
        print("🔄 Loading MiDaS (DPT SwinV2 Tiny 256)...")

        depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-swinv2-tiny-256",
            trust_remote_code=True
        )

        depth_processor = DPTImageProcessor.from_pretrained(
            "Intel/dpt-swinv2-tiny-256",
            trust_remote_code=True
        )

        depth_model.eval()
        print("✅ MiDaS loaded successfully!")

    return depth_model, depth_processor


# ------------------------------------------------
# Decode Base64 Image
# ------------------------------------------------
def decode_base64_image(b64_string):
    img_data = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return pil_img, frame

# ------------------------------------------------
# Simple GET endpoint to check server status
# ------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "Server is running!"}), 200

# ------------------------------------------------
# /detect endpoint
# ------------------------------------------------

@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    print("📩 POST /detect received")

    data = request.json
    if "image_base64" not in data:
        return jsonify({"error": "Missing image_base64"}), 400

    # --- Decode image ---
    pil_img, frame = decode_base64_image(data["image_base64"])

    # --- YOLO detection ---
    yolo = get_yolo()
    results = yolo(frame, conf=0.4, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    # --- Depth estimation ---
    depth_model, depth_processor = get_depth_model()
    inputs = depth_processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        depth = depth_model(**inputs).predicted_depth.squeeze().cpu().numpy()

    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # --- Fusion with Monodepth2-style distance ---
    outputs = []

    # Step 1: Normalize depth map per image
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = depth - depth_min

    # Step 2: Compute per-object distances
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        obj_name = yolo.names[int(cls)]
        crop = depth_normalized[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Take the closest part of the object (5th percentile)
        closest = np.percentile(crop, 5)

        # Convert to approximate meters (scale factor 5.0, tune if needed)
        distance = 5.0 * (1 - closest)

        # Clamp distance to avoid negative or huge values
        distance = max(0.1, min(distance, 10.0))

        outputs.append({
            "object": obj_name,
            "distance": distance,
            "alert": bool(distance < 10.0)
        })

    return jsonify({"alerts": outputs}), 200
# ------------------------------------------------
# Run
# ------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)