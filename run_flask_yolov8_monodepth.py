from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
import cv2
import sys
import torchvision.transforms as transforms

# -----------------------------
# Load YOLOv8s
# -----------------------------
from ultralytics import YOLO
print("🔄 Loading YOLOv8s...")
yolo = YOLO("yolov8s.pt")
print("✅ YOLOv8s loaded!")

# -----------------------------
# Load Monodepth2
# -----------------------------
sys.path.append("monodepth2")
from networks import ResnetEncoder, DepthDecoder

device = torch.device("cpu")

encoder = ResnetEncoder(num_layers=18, pretrained=False, num_input_images=1)
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc)

enc = torch.load("monodepth2/models/mono_640x192/encoder.pth", map_location=device)
encoder.load_state_dict({k: v for k, v in enc.items() if k in encoder.state_dict()})

dec = torch.load("monodepth2/models/mono_640x192/depth.pth", map_location=device)
depth_decoder.load_state_dict(dec)

encoder.eval()
depth_decoder.eval()

transform = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.ToTensor()
])

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/detect", methods=["POST"])
def detect():
    print("📩 Received request")

    data = request.json
    img_data = base64.b64decode(data["image_base64"])
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = frame.shape

    # ---------- YOLOv8 DETECTION ----------
    results = yolo.predict(frame, conf=0.4, imgsz=640, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    # ---------- Monodepth2 DEPTH ----------
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = encoder(input_tensor)
        disp_out = depth_decoder(features)[("disp", 0)]

    disp = disp_out.squeeze().cpu().numpy()
    disp = cv2.resize(disp, (w_img, h_img))

    # ---------- FUSION ----------
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

        # ⚠️ Convert NumPy bool → Python bool
        alert_flag = bool(depth < 10.0)

        output.append({
            "object": object_name,
            "distance": depth,
            "alert": alert_flag   # <-- FIXED HERE
        })

    return jsonify({"alerts": output})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)