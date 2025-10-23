# antispoof/anti_spoof_predict.py

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
from typing import Any

from antispoof.model_lib.MiniFASNet import (
    MiniFASNetV1,
    MiniFASNetV2,
    MiniFASNetV1SE,
    MiniFASNetV2SE,
)
# Try to import MultiFTNet safely; use a typing-safe fallback instead of a local class.
try:
    # type: ignore - some static analyzers may not have access to the optional module
    from antispoof.model_lib.MultiFTNet import MultiFTNet  # type: ignore
    HAS_MULTIFT = True
except Exception:
    # Bind name to None and use typing.Any to avoid introducing a local class
    # that could conflict with the actual imported type during static analysis.
    MultiFTNet = None  # type: Any
    HAS_MULTIFT = False
    HAS_MULTIFT = False

from antispoof.data_io import transform as trans
from antispoof.utility import get_kernel, parse_model_name

# Map model type strings to actual classes
MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}

if HAS_MULTIFT:
    # Wrap MultiFTNet to align with MiniFASNet-like constructor signature
    def MultiFTNet_wrapper(
        embedding_size: int = 128,
        conv6_kernel=(5, 5),
        drop_p: float = 0.0,
        num_classes: int = 3,
        img_channel: int = 3,
    ):
        """
        Wrapper to make MultiFTNet signature compatible with MiniFASNet family.
        drop_p is ignored since MultiFTNet does not use it.
        """
        return MultiFTNet(
            img_channel=img_channel,
            num_classes=num_classes,
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
        )

    MODEL_MAPPING["MultiFTNet"] = MultiFTNet_wrapper


class Detection:
    def __init__(self):
        # Build absolute path to detection model files inside models/detection
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "detection")
        )
        caffemodel = os.path.join(base_dir, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(base_dir, "deploy.prototxt")

        if not os.path.exists(caffemodel) or not os.path.exists(deploy):
            raise FileNotFoundError(
                f"Detection model files not found in {base_dir}. "
                f"Expected deploy.prototxt and Widerface-RetinaFace.caffemodel."
            )

        # Access cv2.dnn safely
        dnn_mod = getattr(cv2, "dnn", None)
        if dnn_mod is None:
            raise RuntimeError("cv2.dnn module not available in this OpenCV build.")

        readNetFromCaffe = getattr(dnn_mod, "readNetFromCaffe", None)
        if readNetFromCaffe is None:
            raise RuntimeError("cv2.dnn.readNetFromCaffe not available.")

        self.detector = readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height

        resize = getattr(cv2, "resize", None)
        INTER_LINEAR = getattr(cv2, "INTER_LINEAR", 1)

        if img.shape[1] * img.shape[0] >= 192 * 192 and resize is not None:
            img = resize(
                img,
                (
                    int(192 * math.sqrt(aspect_ratio)),
                    int(192 / math.sqrt(aspect_ratio)),
                ),
                interpolation=INTER_LINEAR,
            )

        dnn_mod = getattr(cv2, "dnn", None)
        if dnn_mod is None:
            raise RuntimeError("cv2.dnn module not available in this OpenCV build.")

        blobFromImage = getattr(dnn_mod, "blobFromImage", None)
        if blobFromImage is None:
            raise RuntimeError("cv2.dnn.blobFromImage not available.")

        blob = blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, "data")
        out = self.detector.forward("detection_out").squeeze()

        if out.ndim == 1:  # no face detected
            return None

        max_conf_index = np.argmax(out[:, 2])
        if out[max_conf_index, 2] < self.detector_confidence:
            return None

        left = out[max_conf_index, 3] * width
        top = out[max_conf_index, 4] * height
        right = out[max_conf_index, 5] * width
        bottom = out[max_conf_index, 6] * height
        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        if torch.cuda.is_available() and device_id >= 0:
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)

        if model_type not in MODEL_MAPPING:
            raise ValueError(
                f"Model type '{model_type}' not recognized or not supported. "
                f"Available: {list(MODEL_MAPPING.keys())}"
            )

        # Initialize model via mapping (MiniFASNet or MultiFTNet wrapper)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(
            self.device
        )

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = next(keys)
        if first_layer_name.find("module.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([trans.ToTensor()])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result
