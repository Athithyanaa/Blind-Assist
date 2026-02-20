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
import torchvision.transforms as transforms

app = Flask(__name__)

# CORS FIX → allow browser POST + OPTIONS preflight
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ============================================
# Lazy-loaded models (Render RAM protection)
# ============================================
yolo_model = None
encoder = None
depth_decoder = None
md_transform = None


# ========================
# YOLO loader
# ========================
def get_yolo():
    global yolo_model
    if yolo_model is None:
        print("🔄 Loading YOLOv8n...")
        yolo_model = YOLO("yolov8n.pt")
        print("✅ YOLO loaded!")
    return yolo_model


# ========================
# Monodepth2 loader
# ========================
def get_monodepth():
    global encoder, depth_decoder, md_transform

    if encoder is None:
        print("🔄 Loading Monodepth2...")

        sys.path.append("monodepth2")
        from networks import ResnetEncoder, DepthDecoder

        device = torch.device("cpu")

        # Encoder
        encoder = ResnetEncoder(num_layers=18, pretrained=False, num_input_images=1)
        enc = torch.load(
            "monodepth2/models/mono_640x192/encoder.pth",
            map_location=device
        )
        encoder.load_state_dict({k: v for k, v in enc.items() if k in encoder.state_dict()})
        encoder.eval()

        # Decoder
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)
        dec = torch.load(
            "monodepth2/models/mono_640x192/depth.pth",
            map_location=device
        )
        depth_decoder.load_state_dict(dec)
        depth_decoder.eval()

        # Transform
        md_transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor()
        ])

        print("✅ Monodepth2 loaded!")

    return encoder, depth_decoder, md_transform


# ========================
# Base64 → PIL + OpenCV
# ========================
def decode_base64_image(b64_string):
    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return img, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ========================
# /detect endpoint
# ========================
@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    # ===========================
    # Handle CORS preflight
    # ===========================
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    print("📩 POST /detect received")

    data = request.json
    if not data or "image_base64" not in data:
        return jsonify({"error": "Missing image_base64"}), 400

    pil_img, frame = decode_base64_image(data["image_base64"])
    h_img, w_img, _ = frame.shape

    # ---------- YOLO ----------
    yolo = get_yolo()
    results = yolo.predict(frame, conf=0.4, imgsz=640, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    # ---------- Monodepth2 ----------
    encoder, depth_decoder, md_transform = get_monodepth()
    input_tensor = md_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        features = encoder(input_tensor)
        disp_out = depth_decoder(features)[("disp", 0)]

    disp = disp_out.squeeze().cpu().numpy()
    disp = cv2.resize(disp, (w_img, h_img))

    # ---------- Fusion ----------
    output = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        object_name = yolo.names[int(cls)]
        crop = disp[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        avg_disp = float(np.mean(crop))
        depth = float(1 / (avg_disp + 1e-6))
        alert_flag = bool(depth < 10.0)

        output.append({
            "object": object_name,
            "distance": depth,
            "alert": alert_flag
        })

    return jsonify({"alerts": output}), 200


# ========================
# Run server
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)