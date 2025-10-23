from flask import Flask, request, jsonify
import os, base64, datetime, sqlite3, hashlib, cv2, numpy as np
from pathlib import Path
from typing import Any, cast, List, Tuple
import onnxruntime as ort
from antispoof.anti_spoof_predict import AntiSpoofPredict
from antispoof.utility import parse_model_name

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__)

# -------------------------
# Paths
# -------------------------
BASE = Path.cwd()
MODEL_DIR = BASE / "models"
ANTI_SPOOF_MODEL_DIR = MODEL_DIR / "anti_spoof"
ARCFACE_DIR = MODEL_DIR / "arcface"
USER_DB = MODEL_DIR / "users.db"
STATIC_DIR = BASE / "static"
ENROLL_DIR = STATIC_DIR / "enrollment_images"

ENROLL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Config
# -------------------------
MAX_ENROLL_IMAGES = 7
MIN_VALID_IMAGES = 2
MATCH_THRESHOLD = 0.55
SPOOF_CONF_THRESHOLD = 0.9
TARGET_SIZE = (640, 480)


# -------------------------
# Utility Functions
# -------------------------
def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to target size"""
    return cv2.resize(frame, TARGET_SIZE)

def face_sharpness(gray_face: np.ndarray) -> float:
    """Compute sharpness using Laplacian variance"""
    if gray_face is None or gray_face.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

def face_area_ratio(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> float:
    """Compute ratio of face area to frame size"""
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape
    return float((w * h) / (frame_w * frame_h + 1e-9))

def compute_quality_metrics(face_bgr: np.ndarray, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> dict:
    """Return sharpness and area ratio for weighting embeddings"""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharp = face_sharpness(gray)
    area_ratio = face_area_ratio(bbox, frame_shape)
    return {"sharpness": sharp, "area_ratio": area_ratio, "raw_sum": sharp + (area_ratio * 1e4)}

def normalize_weights(metrics_list: List[dict]) -> List[float]:
    """Normalize weights based on quality metrics"""
    raw = np.array([m["raw_sum"] for m in metrics_list], dtype=np.float32)
    raw = np.clip(raw, a_min=1e-6, a_max=50.0)
    exp = np.exp(raw / (np.max(raw) + 1e-6))
    weights = exp / (np.sum(exp) + 1e-9)
    return weights.tolist()

def weighted_mean_and_variance(embs: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute weighted mean and variance of embeddings"""
    embs_arr = np.stack(embs, axis=0)
    weights_arr = np.array(weights, dtype=np.float32).reshape((-1, 1))
    mean = np.sum(embs_arr * weights_arr, axis=0) / (np.sum(weights_arr) + 1e-9)
    diff = embs_arr - mean
    var = np.sum((diff ** 2) * weights_arr, axis=0) / (np.sum(weights_arr) + 1e-9)
    mean /= (np.linalg.norm(mean) + 1e-6)
    return mean.astype(np.float32), var.astype(np.float32)

# -------------------------
# Face Embedding Setup (ArcFace)
# -------------------------
ONNX_ARCFACE_PATH = ARCFACE_DIR / "model.onnx"
ARCFACE_SESSION = None
if ONNX_ARCFACE_PATH.exists():
    ARCFACE_SESSION = ort.InferenceSession(str(ONNX_ARCFACE_PATH), providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, 112, 112).astype(np.float32)
    input_name = ARCFACE_SESSION.get_inputs()[0].name
    ARCFACE_SESSION.run(None, {input_name: dummy})

def preprocess_face_for_arcface(face_bgr: np.ndarray, size=(112, 112)):
    """Preprocess face for ArcFace input"""
    if face_bgr is None or face_bgr.size == 0:
        return None
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, size)
    x = (face.astype(np.float32) - 127.5) / 128.0
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, 0).astype(np.float32)

def get_embedding(face_bgr: np.ndarray):
    """Generate normalized embedding using ArcFace"""
    if ARCFACE_SESSION is None:
        return None
    inp = preprocess_face_for_arcface(face_bgr)
    if inp is None:
        return None
    input_name = ARCFACE_SESSION.get_inputs()[0].name
    out = ARCFACE_SESSION.run(None, {input_name: inp})
    out0_np = np.asarray(out[0])
    emb = out0_np.squeeze().astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-6)
    return emb

# -------------------------
# Anti-Spoofing Setup
# -------------------------
anti_spoof = AntiSpoofPredict(device_id=0)
ANTI_SPOOF_MODELS = [
    os.path.join(ANTI_SPOOF_MODEL_DIR, m)
    for m in os.listdir(ANTI_SPOOF_MODEL_DIR)
    if os.path.isfile(os.path.join(ANTI_SPOOF_MODEL_DIR, m))
]

# -------------------------
# ✅ Admin Add User (Registration)
# -------------------------
@app.route("/admin_add_user", methods=["POST"])
def admin_add_user():
    data = request.get_json()
    user_id = data.get("user_id")
    images = data.get("images")
    
    if not user_id or not images:
        return jsonify({"status": "error", "message": "Missing user_id or images"}), 400

    images = images[:MAX_ENROLL_IMAGES]
    valid_embs, metrics, candidates, reject_reasons = [], [], [], []

    # ---------------------------
    # Process Each Image
    # ---------------------------
    for i, img_data in enumerate(images):
        try:
            img_bytes = base64.b64decode(img_data.split(",")[1])
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame = resize_frame(frame)
        except Exception:
            reject_reasons.append((i + 1, "Invalid image data"))
            continue

        bbox = anti_spoof.get_bbox(frame)
        if bbox is None:
            reject_reasons.append((i + 1, "No face detected"))
            continue

        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]

        # ---------------------------
        # Anti-Spoofing
        # ---------------------------
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
            reject_reasons.append((i + 1, f"Spoof detected (conf={value:.2f})"))
            continue

        # ---------------------------
        # Face Embedding
        # ---------------------------
        emb = get_embedding(face)
        if emb is None:
            reject_reasons.append((i + 1, "Failed to compute embedding"))
            continue

        frame_shape = cast(Tuple[int, int], frame.shape[:2])
        m = compute_quality_metrics(face, (x, y, w, h), frame_shape)
        valid_embs.append(emb)
        metrics.append(m)
        candidates.append((m["raw_sum"], frame))

    # ---------------------------
    # Validate Enrollments
    # ---------------------------
    if len(valid_embs) < MIN_VALID_IMAGES:
        return jsonify({
            "status": "error",
            "message": "Not enough valid images",
            "details": reject_reasons
        }), 400

    # ---------------------------
    # Compute Final Embedding
    # ---------------------------
    weights = normalize_weights(metrics)
    mean_emb, var_emb = weighted_mean_and_variance(valid_embs, weights)
    reg_quality = float(np.mean([m["sharpness"] for m in metrics]))

   # ---------------------------
    # Save Best Image and Return Base64
    # ---------------------------
    candidates.sort(key=lambda x: x[0], reverse=True)
    user_dir = ENROLL_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    best_image = candidates[0][1]
    best_image_path = str(user_dir / "best.jpg")
    cv2.imwrite(best_image_path, best_image)

    # Convert best image to Base64
    _, buffer = cv2.imencode(".jpg", best_image)
    best_image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    # ---------------------------
    # Return Data to Android
    # ---------------------------
    return jsonify({
    "status": "ok",
    "user_id": user_id,
    "embedding": base64.b64encode(mean_emb.tobytes()).decode("utf-8"),
    "variance": base64.b64encode(var_emb.tobytes()).decode("utf-8"),
    "embedding_dim": len(mean_emb),
    "reg_quality": reg_quality,
    "best_image_base64": best_image_base64,

    "rejected_frames": reject_reasons
})


# -------------------------
# Run the App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
