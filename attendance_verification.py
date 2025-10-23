from flask import Flask, request, jsonify
import os, base64, cv2, numpy as np, datetime
from pathlib import Path
import onnxruntime as ort
from antispoof.anti_spoof_predict import AntiSpoofPredict
from antispoof.utility import parse_model_name

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Paths and configs
# -----------------------------
BASE = Path.cwd()
MODEL_DIR = BASE / "models"
ANTI_SPOOF_MODEL_DIR = MODEL_DIR / "anti_spoof"
ARCFACE_DIR = MODEL_DIR / "arcface"

MATCH_THRESHOLD = 0.55
SPOOF_CONF_THRESHOLD = 0.9
TARGET_SIZE = (640, 480)

# -----------------------------
# Load models
# -----------------------------
anti_spoof = AntiSpoofPredict(device_id=0)
ANTI_SPOOF_MODELS = [
    os.path.join(ANTI_SPOOF_MODEL_DIR, m)
    for m in os.listdir(ANTI_SPOOF_MODEL_DIR)
    if os.path.isfile(os.path.join(ANTI_SPOOF_MODEL_DIR, m))
]

ONNX_ARCFACE_PATH = ARCFACE_DIR / "model.onnx"
ARCFACE_SESSION = None
if ONNX_ARCFACE_PATH.exists():
    ARCFACE_SESSION = ort.InferenceSession(str(ONNX_ARCFACE_PATH), providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, 112, 112).astype(np.float32)
    input_name = ARCFACE_SESSION.get_inputs()[0].name
    ARCFACE_SESSION.run(None, {input_name: dummy})

# -----------------------------
# Helper functions
# -----------------------------
def resize_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, TARGET_SIZE)

def preprocess_face_for_arcface(face_bgr: np.ndarray, size=(112, 112)):
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, size)
    x = (face.astype(np.float32) - 127.5) / 128.0
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, 0).astype(np.float32)

def get_embedding(face_bgr: np.ndarray):
    if ARCFACE_SESSION is None:
        return None
    inp = preprocess_face_for_arcface(face_bgr)
    input_name = ARCFACE_SESSION.get_inputs()[0].name
    out = ARCFACE_SESSION.run(None, {input_name: inp})
    emb = np.asarray(out[0]).squeeze().astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-6)
    return emb

def encode_frame_base64(frame):
    """Convert frame (BGR) to Base64 string."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

# -----------------------------
# 🧠 Attendance Verification Endpoint (no DB)
# -----------------------------
@app.route("/verify_attendance", methods=["POST"])
def verify_attendance():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        frame_data = data.get("frame")
        stored_emb_b64 = data.get("embedding")  # from Android DB

        if not user_id or not frame_data or not stored_emb_b64:
            return jsonify({"error": "Missing user_id, frame, or embedding"}), 400

        # Decode Base64 image
        img_bytes = base64.b64decode(frame_data.split(",")[1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame = resize_frame(frame)

        # 1️⃣ Face detection
        bbox = anti_spoof.get_bbox(frame)
        if bbox is None:
            return jsonify({
                "status": "error",
                "message": "No face detected",
                "spoof": "none",
                "match": False
            })

        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]

        # 2️⃣ Anti-spoofing
        prediction = np.zeros((1, 3))
        for model_path in ANTI_SPOOF_MODELS:
            try:
                h_input, w_input, _, _ = parse_model_name(os.path.basename(model_path))
                img_rs = cv2.resize(frame, (w_input, h_input))
                prediction += anti_spoof.predict(img_rs, model_path)
            except Exception:
                continue

        label = int(np.argmax(prediction))
        value = float(prediction[0][label])
        if not (label == 1 and value > SPOOF_CONF_THRESHOLD):
            frame_b64 = encode_frame_base64(frame)
            return jsonify({
                "status": "error",
                "spoof": "spoof",
                "match": False,
                "score": value,
                "suspicious_frame_base64": frame_b64,
                "message": "Spoof detected"
            })

        # 3️⃣ Generate embedding
        emb = get_embedding(face)
        if emb is None:
            return jsonify({"status": "error", "message": "Failed to compute embedding"})

        # 4️⃣ Compare with Android DB embedding
        stored_emb = np.frombuffer(base64.b64decode(stored_emb_b64), dtype=np.float32)
        score = float(np.dot(emb, stored_emb))

        if score < MATCH_THRESHOLD:
            frame_b64 = encode_frame_base64(frame)
            return jsonify({
                "status": "error",
                "spoof": "real",
                "match": False,
                "score": score,
                "suspicious_frame_base64": frame_b64,
                "message": "Face does not match"
            })

        # 5️⃣ Success — return all metrics
        return jsonify({
            "status": "ok",
            "user_id": user_id,
            "match": True,
            "spoof": "real",
            "score": round(score, 3),
            "bbox": [x, y, w, h],
            "message": "Face verified successfully"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
