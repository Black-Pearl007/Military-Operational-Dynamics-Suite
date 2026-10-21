"""
MODLS - Military Operational Dynamics and Logistics Suite (updated)

Changes vs. the old version
---------------------------
* `face_recognition` / `dlib` removed (dlib has no wheels for new Python versions
  and needs a C++ build). Replaced by OpenCV's built-in YuNet (detector) and
  SFace (recognizer) ONNX models. Only `opencv-contrib-python` is needed.
* KCF tracker creation works across OpenCV versions (with a fallback).
* Object tracking no longer uses cv2.selectROI / cv2.imshow (they freeze or crash
  inside Streamlit). The ROI is chosen with sliders and frames render in the page.
* Webcam pages use st.camera_input (works in the browser) plus an optional
  live loop for local use.
* use_column_width -> use_container_width, "[cite: 51]" leftovers removed,
  motion warning shown once per frame instead of once per contour.

Tracker upgrades (this version)
-------------------------------
* YOLOX detector (OpenCV DNN) re-finds the target after it is lost,
* Kalman filter predicts where a hidden target goes and guides the search,
* MobileNetV2 appearance model (re-ID) separates look-alike objects,
* sidebar: camera resolution, processing width and an infrared-camera mode.
Models download once into ./models; if a download fails the tracker falls back to the
older colour/template search.

Install:
    pip install streamlit opencv-contrib-python numpy pandas matplotlib seaborn
Run:
    streamlit run modls_app.py
"""

import os
import tempfile
import threading
import time
import urllib.request

# faster webcam start-up with the Windows Media Foundation backend
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --------------------------------------------------------------------------- #
# Face engine (OpenCV YuNet + SFace) - replaces face_recognition / dlib
# --------------------------------------------------------------------------- #
MODEL_DIR = "models"
FACES_DIR = "registered_faces"
YUNET_FILE = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_FILE = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
COSINE_THRESHOLD = 0.363  # recommended SFace cosine-similarity threshold


def _download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)


@st.cache_resource(show_spinner="Loading face models...")
def get_face_engine():
    """Download (first run only) and load the detector + recognizer."""
    _download(YUNET_URL, YUNET_FILE)
    _download(SFACE_URL, SFACE_FILE)
    detector = cv2.FaceDetectorYN.create(YUNET_FILE, "", (320, 320), 0.8, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(SFACE_FILE, "")
    return detector, recognizer


def detect_faces(detector, frame_bgr):
    """Return an (N, 15) array of faces (x, y, w, h, 10 landmarks, score) or empty."""
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    return faces if faces is not None else np.empty((0, 15), dtype=np.float32)


def embed_face(recognizer, frame_bgr, face_row):
    aligned = recognizer.alignCrop(frame_bgr, face_row)
    return recognizer.feature(aligned)  # shape (1, 128)


def decode_image(uploaded):
    """Streamlit camera_input / file_uploader object -> BGR numpy image."""
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_registered_faces(directory=FACES_DIR):
    """Load saved embeddings: {name: (1,128) float32 array}."""
    faces = {}
    if os.path.isdir(directory):
        for fn in os.listdir(directory):
            if fn.endswith(".npy"):
                faces[fn[:-4]] = np.load(os.path.join(directory, fn))
    return faces


def identify(recognizer, feature, registered):
    """Best match name (or 'Unknown') and its cosine score."""
    best_name, best_score = "Unknown", -1.0
    for name, ref in registered.items():
        score = recognizer.match(feature, ref, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < COSINE_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


def annotate_faces(frame, detector, recognizer, registered):
    for face in detect_faces(detector, frame):
        x, y, w, h = face[:4].astype(int)
        name, score = ("Unknown", 0.0)
        if registered:
            name, score = identify(recognizer, embed_face(recognizer, frame, face), registered)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name} ({score:.2f})" if registered else name,
                    (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return frame


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def home_page():
    st.title("Military Operational Dynamics and Logistics Suite")
    col1, _ = st.columns(2)
    with col1:
        st.header("Welcome to MODLS")
        st.write(
            "MODLS is an innovative application designed to support defense forces in "
            "various tasks including object tracking, cadet face registration and "
            "recognition, armory data management, and intrusion detection."
        )
    st.subheader("Features at a Glance:")
    st.markdown("""
    - **Object Tracking**: Real-time tracking for enhanced situational awareness.
    - **Cadet Face Registration & Recognition**: Secure and quick identification processes.
    - **Armory Data Management**: Efficient tracking and management of armaments.
    - **Data Visualization**: Insightful analytics for strategic decisions.
    - **Intrusion Detection**: Advanced systems to safeguard perimeters.
    """)


TRACKER_FACTORIES = {
    "CSRT": ["TrackerCSRT_create", "legacy.TrackerCSRT_create"],
    "KCF": ["TrackerKCF_create", "legacy.TrackerKCF_create"],
    "MIL": ["TrackerMIL_create", "legacy.TrackerMIL_create"],
}
VIT_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
           "object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx")
VIT_FILE = os.path.join(MODEL_DIR, "object_tracking_vittrack_2023sep.onnx")


@st.cache_resource(show_spinner="Downloading tracker model (first run only)...")
def get_vit_model():
    """Path of the ViT tracker model, or None if it can't be downloaded."""
    try:
        _download(VIT_URL, VIT_FILE)
        return VIT_FILE if os.path.getsize(VIT_FILE) > 100_000 else None
    except Exception:
        return None


def create_tracker(preferred="ViT"):
    """Create the preferred tracker; fall back to whatever this OpenCV build offers."""
    order = [preferred] + [n for n in ("ViT", "CSRT", "KCF", "MIL") if n != preferred]
    for name in order:
        try:
            if name == "ViT":
                model = get_vit_model()
                if model is None:
                    continue
                params = cv2.TrackerVit_Params()
                params.net = model
                return name, cv2.TrackerVit_create(params)
            for path in TRACKER_FACTORIES[name]:
                try:
                    obj = cv2
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    return name, obj()
                except (AttributeError, cv2.error):
                    continue
        except Exception:
            continue
    raise RuntimeError("No OpenCV tracker available. Install opencv-contrib-python.")


# --------------------------------------------------------------------------- #
# Recovery aids for the tracker
#   * YoloxDetector      - finds the object again by itself (YOLOX, 80 COCO classes)
#   * BoxKalman          - predicts where the object goes while it is hidden
#   * AppearanceEmbedder - tells look-alike objects apart (MobileNetV2 features)
# All three download a model (or need none) and degrade gracefully: if a model is
# unavailable the tracker simply falls back to the older colour/template search.
# --------------------------------------------------------------------------- #
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
YOLOX_URL = _ZOO + "object_detection_yolox/object_detection_yolox_2022nov.onnx"
YOLOX_FILE = os.path.join(MODEL_DIR, "object_detection_yolox_2022nov.onnx")
REID_URL = _ZOO + "image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx"
REID_FILE = os.path.join(MODEL_DIR, "image_classification_mobilenetv2_2022apr.onnx")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]  # 80 names, index = YOLOX class id


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp_box(box, shape):
    """Clip an (x, y, w, h) box to the image; returns ints (possibly with w or h == 0)."""
    H, W = shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


class YoloxDetector:
    """YOLOX-s through OpenCV's DNN module (same pre/post-processing as the OpenCV zoo)."""

    SIZE = 640

    def __init__(self, path, conf=0.30, nms=0.45):
        self.net = cv2.dnn.readNet(path)
        self.conf, self.nms = conf, nms
        self._make_anchors()

    def _make_anchors(self):
        grids, strides = [], []
        for s in (8, 16, 32):
            n = self.SIZE // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, frame):
        """Returns a list of (box_xywh, class_id, confidence) in frame coordinates."""
        h, w = frame.shape[:2]
        ratio = min(self.SIZE / h, self.SIZE / w)
        nh, nw = int(h * ratio), int(w * ratio)
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.float32)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(canvas.transpose(2, 0, 1)[None])
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
        return self._decode(np.asarray(out).reshape(-1, out.shape[-1]).copy(), ratio)

    def _decode(self, dets, ratio):
        dets[:, :2] = (dets[:, :2] + self.grids) * self.strides
        dets[:, 2:4] = np.exp(np.clip(dets[:, 2:4], -10, 10)) * self.strides
        scores = dets[:, 4:5] * dets[:, 5:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(scores)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        d, cls, conf = dets[keep], cls[keep], conf[keep]
        boxes = np.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 2], d[:, 3]], 1) / ratio
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), cls.tolist(),
                                      self.conf, self.nms)
        return [(tuple(boxes[i]), int(cls[i]), float(conf[i])) for i in np.array(idx).flatten()]


class AppearanceEmbedder:
    """
    Turns a cropped object into a feature vector (MobileNetV2 output, centred and
    L2-normalised) so two crops can be compared with a dot product (cosine similarity).
    Two different phones of the same model score lower than the same phone seen twice.
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path):
        self.net = cv2.dnn.readNet(path)

    def embed(self, frame, box):
        x, y, w, h = clamp_box(box, frame.shape)
        if w < 8 or h < 8:
            return None
        crop = cv2.cvtColor(cv2.resize(frame[y:y + h, x:x + w], (224, 224)), cv2.COLOR_BGR2RGB)
        blob = ((crop.astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        self.net.setInput(np.ascontiguousarray(blob))
        v = self.net.forward().astype(np.float32).flatten()
        v -= v.mean()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else None


class BoxKalman:
    """
    Constant-velocity Kalman filter on the centre of the box (one step = one frame).
    While the target is hidden the velocity is slowly damped, so the guess drifts to a
    stop instead of flying off the screen.
    """

    VEL_DECAY = 0.98

    def __init__(self, box):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([1.0, 1.0, 0.5, 0.5]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        self.kf = kf
        self.reset(box)

    def reset(self, box):
        x, y, w, h = box
        self.size = (float(w), float(h))
        self.kf.statePost = np.array([[x + w / 2.0], [y + h / 2.0], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

    def predict(self, hidden=False):
        """Advance one frame. Returns (cx, cy, radius): the best guess and its uncertainty."""
        s = self.kf.predict()
        if hidden:
            post = np.array(self.kf.statePost, np.float32)
            post[2:4] *= self.VEL_DECAY
            self.kf.statePost = post
        p = np.array(self.kf.errorCovPost)
        radius = max(1.5 * max(self.size), 3.0 * float(np.sqrt(max(p[0, 0], 0.0))))
        return float(s[0, 0]), float(s[1, 0]), radius

    def correct(self, box):
        x, y, w, h = box
        self.kf.correct(np.array([[x + w / 2.0], [y + h / 2.0]], np.float32))
        self.size = (float(w), float(h))


def _load_yolox():
    _download(YOLOX_URL, YOLOX_FILE)
    return YoloxDetector(YOLOX_FILE)


def _load_reid():
    _download(REID_URL, REID_FILE)
    return AppearanceEmbedder(REID_FILE)


@st.cache_resource(show_spinner="Downloading YOLO detector (first run only, ~35 MB)...")
def _cached_yolox():
    return _load_yolox()          # exceptions are not cached, so a failed download is retried


@st.cache_resource(show_spinner="Downloading recognition model (first run only)...")
def _cached_reid():
    return _load_reid()


def get_yolox():
    try:
        return _cached_yolox()
    except Exception as exc:
        st.warning(f"YOLO detector unavailable ({exc.__class__.__name__}); "
                   "using template search instead.")
        return None


def get_reid():
    try:
        return _cached_reid()
    except Exception as exc:
        st.warning(f"Recognition model unavailable ({exc.__class__.__name__}); "
                   "using colour matching instead.")
        return None


class RobustTracker:
    """
    Wraps an OpenCV tracker and adds what the raw trackers lack:
      * a LOST state - ViT reports its own confidence score; for every tracker the
        centre of the box is also compared (colour histogram) with the locked target,
        so the box is not left on your head / the background when the target leaves,
      * Kalman prediction - while the target is hidden, a filter keeps estimating where
        it should be (`predicted_box`), and that estimate guides the search,
      * re-acquisition - while lost, a YOLO detector proposes every object of the same
        class; each candidate is scored on appearance (recognition model), colour and
        distance from the predicted position, and tracking resumes only when ONE
        candidate clearly matches (so a look-alike is not mistaken for the target).
        Without YOLO/recognition the older multi-scale template search is used.
    """

    MIN_SCORE = 0.30      # ViT confidence below this = not the target
    LOST_DIST = 0.70      # colour distance (0 same .. 1 different) above this = not the target
    LOST_FRAMES = 6       # consecutive bad frames before declaring LOST
    MATCH_SCORE = 0.55    # template-match score needed to re-acquire (fallback search)
    REACQ_DIST = 0.50     # colour distance allowed for a re-acquired box (fallback search)

    DETECT_EVERY = 4      # run YOLO every Nth frame while lost (it is the slow part)
    MAX_PREDICT = 45      # frames the Kalman guess is shown/used (~2 s), then search everywhere
    MARGIN = 0.06         # best candidate must beat the runner-up by this much
    GALLERY = 8           # appearance samples kept for the locked target
    LEARN_EVERY = 15      # add a fresh appearance sample every N good frames
    W_EMB, W_COL, W_POS = 0.5, 0.3, 0.2

    def __init__(self, name, frame, roi, detector=None, embedder=None,
                 use_kalman=True, ir_mode=False, accept=0.55):
        self.factory_name = name
        self.detector, self.embedder = detector, embedder
        self.ir, self.accept = ir_mode, accept
        self.w_col = self.W_COL * (0.5 if ir_mode else 1.0)   # colour says little in IR
        self._start(frame, roi)
        x, y, w, h = self.box
        patch = frame[y:y + h, x:x + w]
        self.hist = self._hist(patch)
        self.template = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self.lost, self.bad, self.n, self.lost_for = False, 0, 0, 0
        self.reacquired_by = None                        # "detector" / "template" after a recovery

        self.kalman = BoxKalman(self.box) if use_kalman else None
        self.pred_center, self.pred_radius, self.predicted_box = None, 0.0, None

        self.gallery = []
        if self.embedder is not None:
            e = self.embedder.embed(frame, self.box)
            if e is not None:
                self.gallery.append(e)

        self.cls, self.cls_name = None, None
        if self.detector is not None:
            self._learn_class(frame)

    # ----- setup helpers --------------------------------------------------- #
    def _start(self, frame, roi):
        self.name, self.tracker = create_tracker(self.factory_name)
        self.box = tuple(int(v) for v in roi)
        self.tracker.init(frame, self.box)
        self.score = 1.0

    def _learn_class(self, frame):
        """Which object class is the target? (the detection that overlaps the box most)"""
        best, best_iou = None, 0.3
        for box, cls, _conf in self.detector.detect(frame):
            i = box_iou(box, self.box)
            if i > best_iou:
                best, best_iou = cls, i
        if best is not None:
            self.cls, self.cls_name = best, COCO_CLASSES[best]

    def _hist(self, patch):
        """Histogram of the CENTRE of the box (ignores background at the edges).
        Colour (HSV) normally; plain brightness for infrared / grayscale cameras."""
        h, w = patch.shape[:2]
        patch = patch[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        if patch.size == 0:
            return None
        if self.ir:
            hist = cv2.calcHist([cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
        else:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2].astype(np.float32)
            gain = 128.0 / max(float(np.median(v)), 25.0)    # cancel overall brightness changes
            hsv[:, :, 2] = np.clip(v * gain, 0, 255).astype(np.uint8)  # (shadow / dimmer light)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 8], [0, 180, 0, 256, 0, 256])
        return (hist / max(hist.sum(), 1e-6)).astype(np.float32)

    def _distance(self, frame, box):
        x, y, w, h = clamp_box(box, frame.shape)
        hist = self._hist(frame[y:y + h, x:x + w])
        if hist is None or self.hist is None:
            return 1.0
        return cv2.compareHist(self.hist, hist, cv2.HISTCMP_BHATTACHARYYA)

    # ----- per-frame update ------------------------------------------------ #
    def update(self, frame):
        """Returns (found, box). found is False while the target is lost/hidden."""
        self.n += 1
        if self.kalman is not None:
            cx, cy, rad = self.kalman.predict(hidden=self.lost)
        if not self.lost:
            ok, box = self.tracker.update(frame)
            if ok and self.name == "ViT":
                self.score = float(self.tracker.getTrackingScore())
                ok = self.score >= self.MIN_SCORE
            if ok:
                box = tuple(int(v) for v in box)
                if box[2] > 4 and box[3] > 4:
                    dist = self._distance(frame, box)
                    if dist <= self.LOST_DIST:
                        self.bad, self.box, self.predicted_box = 0, box, None
                        if self.kalman is not None:
                            self.kalman.correct(box)
                        self._learn_appearance(frame, box, dist)
                        return True, box
            self.bad += 1
            if self.bad < self.LOST_FRAMES:
                return True, self.box          # brief glitch: keep last good box
            self.lost, self.lost_for = True, 0
        self.lost_for += 1
        self._update_prediction(frame, (cx, cy, rad) if self.kalman is not None else None)
        self._reacquire(frame)
        return (not self.lost), self.box

    def _update_prediction(self, frame, pred):
        self.pred_center, self.predicted_box = None, None
        if pred is None or self.lost_for > self.MAX_PREDICT:
            return
        cx, cy, rad = pred
        self.pred_center, self.pred_radius = (cx, cy), rad
        H, W = frame.shape[:2]
        if 0 <= cx < W and 0 <= cy < H:                  # only draw it while on screen
            w, h = self.kalman.size
            self.predicted_box = tuple(int(v) for v in (cx - w / 2, cy - h / 2, w, h))

    def _learn_appearance(self, frame, box, dist):
        """Keep a few fresh, trustworthy views of the target (it changes with angle/light)."""
        if self.embedder is None or self.n % self.LEARN_EVERY or dist > 0.45:
            return
        if self.name == "ViT" and self.score < 0.6:
            return
        e = self.embedder.embed(frame, box)
        if e is None:
            return
        self.gallery.append(e)
        if len(self.gallery) > self.GALLERY:
            del self.gallery[1]                          # keep the original sample (index 0)

    # ----- re-acquisition -------------------------------------------------- #
    def _score(self, frame, box):
        """0..1 'is this my target?' - appearance + colour + closeness to the Kalman guess."""
        parts = [(self.w_col, 1.0 - self._distance(frame, box))]
        if self.embedder is not None and self.gallery:
            e = self.embedder.embed(frame, box)
            if e is not None:
                sim = max(float(e @ g) for g in self.gallery)
                parts.append((self.W_EMB, min(max((sim - 0.3) / 0.7, 0.0), 1.0)))
        if self.pred_center is not None:
            bx, by, bw, bh = box
            d = float(np.hypot(bx + bw / 2 - self.pred_center[0], by + bh / 2 - self.pred_center[1]))
            parts.append((self.W_POS, float(np.exp(-(d / max(self.pred_radius, 1.0)) ** 2))))
        total = sum(w for w, _ in parts)
        return sum(w * v for w, v in parts) / total

    def _resume(self, frame, box, how):
        self.reacquired_by = how
        self._start(frame, box)
        self.lost, self.bad, self.lost_for = False, 0, 0
        self.pred_center, self.predicted_box = None, None
        if self.kalman is not None:
            self.kalman.reset(self.box)

    def _reacquire(self, frame):
        if self.detector is not None and self.cls is not None:
            if self.n % self.DETECT_EVERY == 0 and self._reacquire_with_detector(frame):
                return
        if self.n % 2 == 0:                    # cheap fallback, every 2nd frame
            self._reacquire_with_template(frame)

    def _reacquire_with_detector(self, frame):
        cands = []
        for box, cls, _conf in self.detector.detect(frame):
            if cls != self.cls:
                continue
            box = clamp_box(box, frame.shape)
            if box[2] > 8 and box[3] > 8:
                cands.append((self._score(frame, box), box))
        if not cands:
            return False
        cands.sort(key=lambda c: c[0], reverse=True)
        best_score, best_box = cands[0]
        if best_score < self.accept:
            return False                                 # nothing resembles the target enough
        if len(cands) > 1 and best_score - cands[1][0] < self.MARGIN:
            return False                                 # two look-alikes: too risky to guess
        self._resume(frame, best_box, "detector")
        return True

    def _reacquire_with_template(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        best_score, best_box = -1.0, None
        for rot in (None, cv2.ROTATE_90_CLOCKWISE):      # also try the target turned 90 deg
            tpl = self.template if rot is None else cv2.rotate(self.template, rot)
            for scale in (0.6, 0.8, 1.0, 1.25, 1.5):
                tw, th = int(tpl.shape[1] * scale * 0.5), int(tpl.shape[0] * scale * 0.5)
                if tw < 8 or th < 8 or tw >= small.shape[1] or th >= small.shape[0]:
                    continue
                res = cv2.matchTemplate(small, cv2.resize(tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score, best_box = score, (loc[0] * 2, loc[1] * 2, tw * 2, th * 2)
        if best_box is None or best_score < self.MATCH_SCORE:
            return
        if self._distance(frame, best_box) > self.REACQ_DIST:
            return                                       # similar shape, wrong colour
        if self.embedder is not None and self.gallery and self._score(frame, best_box) < self.accept:
            return                                       # looks like a different object
        self._resume(frame, best_box, "template")


def draw_tracker_status(frame, tracker, found, box):
    """Shared drawing for the live and uploaded-video pages."""
    if found:
        bx, by, bw, bh = [int(v) for v in box]
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        label = f"TRACKING ({tracker.name}"
        label += f" {tracker.score:.2f})" if tracker.name == "ViT" else ")"
        if tracker.cls_name:
            label += f" - {tracker.cls_name}"
        cv2.putText(frame, label, (bx, max(by - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return
    pb = tracker.predicted_box
    if pb is not None:                                   # Kalman guess while hidden
        px, py, pw, ph = pb
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
        cv2.putText(frame, "PREDICTED", (px, max(py - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, "TARGET HIDDEN - searching... (or press Reset)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def describe_aids(tracker):
    """One-line summary of which recovery aids are active for this tracker."""
    parts = []
    if tracker.detector is not None:
        parts.append(f"YOLO re-acquisition (target class: {tracker.cls_name})" if tracker.cls_name
                     else "YOLO loaded, but the target is not a known COCO class - "
                          "using template search")
    if tracker.kalman is not None:
        parts.append("Kalman prediction")
    if tracker.embedder is not None:
        parts.append("appearance re-ID")
    if tracker.ir:
        parts.append("infrared mode")
    return "Recovery aids: " + (", ".join(parts) if parts else "none (template search only)")


def make_tracker(choice, frame, roi):
    """Build a RobustTracker using the recovery aids ticked on the Object Tracking page."""
    s = st.session_state
    detector = get_yolox() if s.get("trk_yolo", True) else None
    embedder = get_reid() if s.get("trk_reid", True) else None
    return RobustTracker(choice, frame, roi, detector=detector, embedder=embedder,
                         use_kalman=s.get("trk_kalman", True), ir_mode=s.get("ir_mode", False),
                         accept=float(s.get("trk_strict", 0.55)))


def show_image(target, image, **kwargs):
    """st.image that works on old and new Streamlit (use_container_width was removed)."""
    try:
        target.image(image, width="stretch", **kwargs)
    except TypeError:
        target.image(image, use_container_width=True, **kwargs)


CAM_HELP = ("Webcam failed to deliver frames. Close other apps that use the camera "
            "(Zoom, Teams, Camera app, other browser tabs), or try another 'Camera index' "
            "in the sidebar.")


def parse_resolution(text):
    w, h = str(text).lower().split("x")
    return int(w), int(h)


def open_capture(index=0, size=(640, 480)):
    """Open a webcam, trying several Windows backends (DSHOW / MSMF / default)."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            for _ in range(20):  # some cameras need a moment before the first frame
                ok, _frame = cap.read()
                if ok:
                    return cap
                time.sleep(0.1)
        cap.release()
    return None


class CameraStream:
    """
    One background thread owns the webcam and always keeps the newest frame.
    Streamlit reruns (button clicks, slider moves) never open/close the device,
    which is what caused 'webcam failed' and native crashes before.
    """

    WARMUP_FRAMES = 10  # let auto-exposure settle

    def __init__(self, index=0, size=(640, 480)):
        self.index, self.size = index, size
        self._cap = open_capture(index, size)
        self._lock = threading.Lock()
        self._frame, self._id = None, 0
        self.running = self._cap is not None
        if self.running:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        fails, count = 0, 0
        while self.running:
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                if fails > 300:  # camera unplugged / taken by another app
                    break
                time.sleep(0.01)
                continue
            fails, count = 0, count + 1
            if count > self.WARMUP_FRAMES:
                with self._lock:
                    self._frame, self._id = frame, self._id + 1
        self.running = False
        self._cap.release()  # released by the same thread that reads it

    def read(self, last_id=-1, timeout=5.0):
        """Wait for a frame newer than last_id. Returns (id, frame) or (last_id, None)."""
        end = time.time() + timeout
        while time.time() < end and self.running:
            with self._lock:
                if self._frame is not None and self._id != last_id:
                    return self._id, self._frame.copy()
            time.sleep(0.004)
        return last_id, None

    def stop(self):
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=3)


class SharedCam:
    """cv2.VideoCapture-like wrapper (read/release) around the shared CameraStream."""

    def __init__(self, stream):
        self.stream, self.last = stream, -1

    def read(self):
        self.last, frame = self.stream.read(self.last)
        return frame is not None, frame

    def release(self):
        pass  # the stream stays alive across Streamlit reruns; see release_camera()


@st.cache_resource
def _camera_registry():
    return {"lock": threading.Lock(), "stream": None}


def open_camera():
    index = int(st.session_state.get("cam_index", 0))
    size = parse_resolution(st.session_state.get("cam_res", "640x480"))
    reg = _camera_registry()
    with reg["lock"]:
        stream = reg["stream"]
        if stream is None or not stream.running or stream.index != index or stream.size != size:
            if stream is not None:
                stream.stop()
            stream = reg["stream"] = CameraStream(index, size)
        if not stream.running:
            reg["stream"] = None
            return None
        return SharedCam(stream)


def release_camera():
    """Turn the webcam off (called when a page's checkbox is unticked / page changes)."""
    reg = _camera_registry()
    with reg["lock"]:
        if reg["stream"] is not None:
            reg["stream"].stop()
            reg["stream"] = None


class Pacer:
    """Cap the display rate so the browser isn't flooded with frames."""

    def __init__(self, fps):
        self.dt, self.t = 1.0 / fps, time.time()

    def wait(self):
        delay = self.t + self.dt - time.time()
        if delay > 0:
            time.sleep(delay)
        self.t = time.time()


def read_frame(cap, mirror=True, width=None):
    width = width or int(st.session_state.get("proc_width", 640))
    ok, frame = cap.read()
    if not ok:
        return None
    if mirror:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)))
    return frame


def save_upload(uploaded):
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded.read())
        return tfile.name


def roi_sliders(W, H, key, disabled=False):
    c1, c2 = st.columns(2)
    x = c1.slider("X", 0, W - 1, W // 3, key=f"{key}_x", disabled=disabled)
    y = c1.slider("Y", 0, H - 1, H // 3, key=f"{key}_y", disabled=disabled)
    w = c2.slider("Width", 1, W, max(1, W // 3), key=f"{key}_w", disabled=disabled)
    h = c2.slider("Height", 1, H, max(1, H // 3), key=f"{key}_h", disabled=disabled)
    return x, y, min(w, W - x), min(h, H - y)


def _reset_tracking():
    st.session_state["trk_armed"] = False
    st.session_state.pop("trk_snap", None)


def _lock_tracking():
    st.session_state["trk_armed"] = True


def _request_pick():
    st.session_state["trk_pick"] = True


def select_target_with_mouse(frame):
    """Pop-up OpenCV window: drag a box around the target, press ENTER. None = no GUI support."""
    win = "Select target: drag a box, press ENTER (C = cancel)"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        roi = cv2.selectROI(win, frame, showCrosshair=False, fromCenter=False)
        cv2.destroyWindow(win)
        cv2.waitKey(1)
    except cv2.error:
        return None
    return tuple(int(v) for v in roi) if roi[2] > 5 and roi[3] > 5 else ()


def live_tracking(tracker_choice):
    run = st.checkbox("Camera on", key="trk_run")
    if not run:
        release_camera()
        st.info("Tick 'Camera on' to start the webcam. Untick it to stop.")
        return
    cap = open_camera()
    if cap is None:
        st.error("Could not open the webcam. Close other apps using it and try again.")
        return
    try:
        frame = read_frame(cap)
        if frame is None:
            st.error(CAM_HELP)
            return
        H, W = frame.shape[:2]

        # --- choose the target with the mouse (pop-up window) ---
        if st.session_state.pop("trk_pick", False):
            picked = select_target_with_mouse(frame)
            if picked is None:
                st.error("The pop-up selector needs the full 'opencv-contrib-python' package "
                         "(not the headless one). Use the sliders instead.")
            elif picked:
                st.session_state["trk_snap"] = (frame.copy(), picked)
                st.session_state["trk_armed"] = True
        armed = st.session_state.get("trk_armed", False)

        st.caption("Best: press 'Select target with mouse', drag a tight box around the object "
                   "and press ENTER. Or use the sliders (the video is mirrored), press Lock and "
                   "hold the object in the box for 3 seconds.")
        roi = roi_sliders(W, H, "trk_live", disabled=armed)
        b1, b2, b3 = st.columns(3)
        b1.button("Select target with mouse", on_click=_request_pick, disabled=armed)
        b2.button("Lock slider box & track", on_click=_lock_tracking, disabled=armed)
        b3.button("Reset / new target", on_click=_reset_tracking)

        note, stframe = st.empty(), st.empty()
        tracker = None
        t_lock, t_prev, fps = time.time(), time.time(), 0.0
        pacer = Pacer(20)
        while True:
            frame = read_frame(cap)
            if frame is None:
                st.error(CAM_HELP)
                break
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            x, y, w, h = roi

            if not armed:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Select the target (mouse button or sliders)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            elif tracker is None:
                snap = st.session_state.pop("trk_snap", None)
                if snap is not None:                       # chosen with the mouse
                    tracker = make_tracker(tracker_choice, snap[0], snap[1])
                else:                                      # chosen with sliders: 3 s countdown
                    remaining = 3 - (now - t_lock)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, f"Hold the object in the box... {max(0, int(remaining) + 1)}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if remaining <= 0:
                        tracker = make_tracker(tracker_choice, frame, (x, y, w, h))
                if tracker is not None and tracker.name != tracker_choice:
                    note.warning(f"{tracker_choice} is not available (model download needs "
                                 f"internet once) - using {tracker.name} instead.")
                elif tracker is not None:
                    note.caption(describe_aids(tracker))
            else:
                found, box = tracker.update(frame)
                draw_tracker_status(frame, tracker, found, box)
            cv2.putText(frame, f"{fps:.0f} FPS", (W - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            show_image(stframe, frame, channels="BGR", output_format="JPEG")
            pacer.wait()
    finally:
        cap.release()


def uploaded_tracking(tracker_choice):
    uploaded = st.file_uploader("Upload a video for object tracking",
                                type=["mp4", "mov", "avi", "asf", "m4v"])
    if uploaded is None:
        return
    path = save_upload(uploaded)
    cap = cv2.VideoCapture(path)
    ok, first = cap.read()
    cap.release()
    if not ok:
        st.error("Failed to read video file.")
        return

    H, W = first.shape[:2]
    st.write("Select the object to track (first frame preview):")
    x, y, w, h = roi_sliders(W, H, "trk_upload")
    preview = first.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
    show_image(st, preview, channels="BGR")

    if st.button("Start Tracking"):
        tracker = make_tracker(tracker_choice, first, (x, y, w, h))
        st.caption(f"Tracker: {tracker.name}. {describe_aids(tracker)}")
        cap = cv2.VideoCapture(path)
        cap.read()
        stframe = st.empty()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            found, box = tracker.update(frame)
            draw_tracker_status(frame, tracker, found, box)
            show_image(stframe, frame, channels="BGR")
        cap.release()
        st.success("Tracking finished.")


def object_tracking_page():
    st.title("Object Tracking")
    mode = st.radio("Source", ["Live webcam", "Upload video"], horizontal=True, key="trk_mode")
    tracker_choice = st.selectbox(
        "Tracker algorithm", ["ViT", "CSRT", "KCF", "MIL"],
        format_func=lambda n: {"ViT": "ViT (deep learning - most robust)",
                               "CSRT": "CSRT (accurate classic)",
                               "KCF": "KCF (fastest, weakest)",
                               "MIL": "MIL (fallback)"}[n],
        help="ViT is a small neural-network tracker and the most robust. It downloads a "
             "~1 MB model on first use; if that fails the app falls back to CSRT.")
    with st.expander("Recovery aids - help the tracker find the target again "
                     "(set these BEFORE locking a target)"):
        st.checkbox("YOLO detector: find the object again by itself after losing it",
                    value=True, key="trk_yolo",
                    help="YOLOX proposes every object of the target's class (e.g. 'cell phone'). "
                         "Downloads a ~35 MB model once. Only works for the 80 COCO classes.")
        st.checkbox("Kalman filter: predict where the object goes while it is hidden",
                    value=True, key="trk_kalman",
                    help="Shows a yellow PREDICTED box and prefers candidates near it.")
        st.checkbox("Recognition model: tell apart similar-looking objects",
                    value=True, key="trk_reid",
                    help="Compares appearance features so a second, similar object is not "
                         "mistaken for your target.")
        st.slider("Re-identification strictness", 0.30, 0.90, 0.55, 0.05, key="trk_strict",
                  help="Higher = only resume tracking when very sure (fewer wrong locks, "
                       "slower recovery). Lower = recover faster but may pick a look-alike.")
    if mode == "Live webcam":
        live_tracking(tracker_choice)
    else:
        uploaded_tracking(tracker_choice)


def cadet_face_registration_page():
    st.title("Cadet Face Registration")
    os.makedirs(FACES_DIR, exist_ok=True)
    detector, recognizer = get_face_engine()

    name = st.text_input("Enter Cadet's Name").strip()
    source = st.radio("Image source", ["Webcam snapshot", "Upload photo"], horizontal=True)
    image = (st.camera_input("Take a photo") if source == "Webcam snapshot"
             else st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"]))

    if st.button("Register Face"):
        if not name:
            st.error("Please enter the cadet's name.")
            return
        if "/" in name or "\\" in name or ".." in name:
            st.error("Invalid characters in name.")
            return
        if image is None:
            st.error("Please provide an image first.")
            return

        frame = decode_image(image)
        faces = detect_faces(detector, frame)
        if len(faces) == 0:
            st.error("No faces detected. Please try again.")
            return
        if len(faces) > 1:
            st.warning("Multiple faces found - using the largest one.")
        face = max(faces, key=lambda f: f[2] * f[3])

        np.save(os.path.join(FACES_DIR, f"{name}.npy"), embed_face(recognizer, frame, face))
        x, y, w, h = np.clip(face[:4].astype(int), 0, None)
        cv2.imwrite(os.path.join(FACES_DIR, f"{name}.jpg"), frame[y:y + h, x:x + w])
        st.success(f"Registered {name}'s face successfully.")

    registered = sorted(load_registered_faces())
    if registered:
        st.caption("Registered cadets: " + ", ".join(registered))


def cadet_face_recognition_page():
    st.title("Cadet Face Recognition")
    detector, recognizer = get_face_engine()
    registered = load_registered_faces()
    if not registered:
        st.warning("No registered faces yet. Register a cadet first.")

    mode = st.radio("Mode", ["Webcam snapshot", "Live webcam (local only)"], horizontal=True)

    if mode == "Webcam snapshot":
        shot = st.camera_input("Take a photo to identify")
        if shot is not None:
            frame = annotate_faces(decode_image(shot), detector, recognizer, registered)
            show_image(st, frame, channels="BGR")
    else:
        run = st.checkbox("Start Face Recognition", key="fr_run")
        stframe = st.empty()
        if not run:
            release_camera()
            return
        cap = open_camera()
        if cap is None:
            st.error("Could not open the webcam. Close other apps using it and try again.")
            return
        pacer = Pacer(12)
        try:
            while True:
                frame = read_frame(cap)
                if frame is None:
                    st.error(CAM_HELP)
                    break
                frame = annotate_faces(frame, detector, recognizer, registered)
                show_image(stframe, frame, channels="BGR", output_format="JPEG")
                pacer.wait()
        finally:
            cap.release()


# --------------------------------------------------------------------------- #
# Armory data
# --------------------------------------------------------------------------- #
def load_data(filename="armory_data.csv"):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame({
        "Weapon Name": ["AK-47", "M1911", "M4A1", "M16", "Glock"],
        "Quantity Available": [100, 50, 75, 30, 60],
        "Condition": ["Excellent", "Good", "Fair", "Good", "Poor"],
        "Caliber": ["7.62x39mm", "45 ACP", "5.56x45mm", "5.56x45mm", "9mm"],
        "Manufacturer": ["Izhmash", "Colt", "Colt", "Colt", "Glock"],
        "Type": ["Assault Rifle", "Handgun", "Assault Rifle", "Assault Rifle", "Handgun"],
        "Country of Origin": ["Russia", "USA", "USA", "USA", "Austria"],
        "Year of Manufacture": [1950, 1911, 1990, 1985, 2000],
        "Weight (kg)": [4.3, 1.1, 3.0, 3.5, 0.7],
        "Effective Range (m)": [400, 50, 500, 600, 50],
    })


def armory_data_registration_page():
    st.title("Armory Data Registration")
    st.header("Add New Weapon Data")
    weapon_name = st.text_input("Weapon Name")
    quantity = st.number_input("Quantity Available", min_value=0, step=1)
    condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
    caliber = st.text_input("Caliber")
    manufacturer = st.text_input("Manufacturer")
    weapon_type = st.selectbox(
        "Type", ["Assault Rifle", "Handgun", "Submachine Gun", "Shotgun", "Sniper Rifle"])
    country = st.text_input("Country of Origin")
    manufacture_year = st.number_input("Year of Manufacture", min_value=1800,
                                       max_value=2026, value=2000, step=1)
    weight = st.number_input("Weight (kg)", min_value=0.01, step=0.01)
    effective_range = st.number_input("Effective Range (m)", min_value=1, step=1)

    if st.button("Add Data"):
        if not weapon_name.strip():
            st.error("Weapon Name is required.")
            return
        data = load_data()
        new_data = pd.DataFrame([{
            "Weapon Name": weapon_name, "Quantity Available": quantity,
            "Condition": condition, "Caliber": caliber, "Manufacturer": manufacturer,
            "Type": weapon_type, "Country of Origin": country,
            "Year of Manufacture": manufacture_year, "Weight (kg)": weight,
            "Effective Range (m)": effective_range,
        }])
        data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv("armory_data.csv", index=False)
        st.success("Data Added Successfully!")


def armory_data_visualization_page():
    st.title("Armory Data Visualization")
    data = load_data()
    selected_column = st.selectbox("Select a Column for Visualization", data.columns)
    col1, col2 = st.columns(2)
    # pandas 3 uses a "str" dtype, so test for numeric instead of dtype == 'object'
    if not pd.api.types.is_numeric_dtype(data[selected_column]):
        with col1:
            st.subheader("Data Distribution")
            st.bar_chart(data[selected_column].value_counts())
        with col2:
            st.subheader("Data Table")
            st.dataframe(data)
    else:
        with col1:
            st.subheader("Histogram")
            fig, ax = plt.subplots(figsize=(8, 6))
            data[selected_column].plot(kind="hist", bins=20, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            st.subheader("Box Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, y=selected_column, ax=ax)
            st.pyplot(fig)
            plt.close(fig)


# --------------------------------------------------------------------------- #
# Intrusion detection / night vision
# --------------------------------------------------------------------------- #
def detect_motion(prev, cur, threshold, min_area):
    diff = cv2.absdiff(prev, cur)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def motion_stream(cap, threshold, min_area, save_snapshots=False, fps=None):
    """Run motion detection on frames from `cap` until it ends / the page reruns."""
    stframe, alert, counter = st.empty(), st.empty(), st.empty()
    ok, prev = cap.read()
    if not ok:
        st.error("Failed to read from the video source.")
        return
    events, last_event = 0, 0.0
    pacer = Pacer(fps) if fps else None
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        boxes = detect_motion(prev, cur, threshold, min_area)
        out = cur.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, stamp, (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if boxes:
            alert.error("INTRUSION DETECTED!")
            now = time.time()
            if now - last_event > 3:  # count one event per 3 seconds
                events += 1
                last_event = now
                if save_snapshots:
                    os.makedirs("intrusions", exist_ok=True)
                    cv2.imwrite(os.path.join(
                        "intrusions", time.strftime("%Y%m%d_%H%M%S") + ".jpg"), out)
        else:
            alert.success("Area clear")
        counter.caption(f"Intrusion events this session: {events}")
        show_image(stframe, out, channels="BGR", output_format="JPEG")
        if pacer:
            pacer.wait()
        prev = cur


def intrusion_detection_page():
    st.title("Intrusion Detection")
    mode = st.radio("Source", ["Live webcam", "Upload video"], horizontal=True, key="int_mode")
    motion_threshold = st.slider("Motion Detection Threshold", 25, 100, 50)
    min_area = st.slider("Minimum motion area (pixels)", 200, 5000, 1000, step=100)

    if mode == "Live webcam":
        save = st.checkbox("Save a snapshot to the 'intrusions' folder on each event", value=True)
        run = st.checkbox("Start monitoring", key="int_run")
        if not run:
            release_camera()
            st.info("Tick 'Start monitoring' to begin. Untick it to stop.")
            return
        cap = open_camera()
        if cap is None:
            st.error("Could not open the webcam. Close other apps using it and try again.")
            return
        try:
            motion_stream(cap, motion_threshold, min_area, save_snapshots=save, fps=20)
        finally:
            cap.release()
    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])
        if uploaded is not None:
            cap = cv2.VideoCapture(save_upload(uploaded))
            try:
                motion_stream(cap, motion_threshold, min_area)
            finally:
                cap.release()


_VIGNETTE_CACHE = {}


def _vignette(h, w):
    if (h, w) not in _VIGNETTE_CACHE:
        ys, xs = np.ogrid[:h, :w]
        d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
        _VIGNETTE_CACHE[(h, w)] = np.clip(1.15 - 0.55 * d ** 2, 0.25, 1.0).astype(np.float32)
    return _VIGNETTE_CACHE[(h, w)]


def apply_night_vision_effect(frame, state, max_gain=12.0, smoothing=12):
    """
    Low-light enhancement:
      1. motion-adaptive temporal smoothing: still areas are averaged over ~`smoothing`
         frames (noise drops a lot) while moving areas follow the current frame, so
         moving objects stay sharp instead of smearing,
      2. auto-stretch of the histogram (adaptive gain up to `max_gain`x),
      3. gamma lift + CLAHE + sharpening,
      4. green phosphor tint with vignette.
    `state` is a dict kept by the caller between frames.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    avg = state.get("avg")
    if avg is None or avg.shape != g.shape:
        avg = g.copy()
    d = np.abs(cv2.GaussianBlur(g, (7, 7), 0) - cv2.GaussianBlur(avg, (7, 7), 0))
    sigma = max(float(np.median(d)) / 0.6745, 0.15)      # noise level estimate
    t0 = min(3 * sigma, 4.0)
    w = np.clip((d - t0) / t0, 0, 1)                      # 0 = still, 1 = moving
    alpha = 1.0 / smoothing + (1.0 - 1.0 / smoothing) * w
    avg += alpha * (g - avg)
    state["avg"] = avg

    lo, hi = np.percentile(avg, (1, 99.5))
    span = max(hi - lo, 255.0 / max_gain)                # never amplify more than max_gain
    img = np.clip((avg - lo) / span, 0, 1) ** 0.7
    u8 = (img * 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (3, 3), 0)
    u8 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(u8)
    blur = cv2.GaussianBlur(u8, (0, 0), 2)
    u8 = cv2.addWeighted(u8, 1.4, blur, -0.4, 0)

    lum = u8.astype(np.float32) * _vignette(*u8.shape)
    out = np.zeros_like(frame, dtype=np.float32)
    out[:, :, 1] = lum                  # green
    out[:, :, 2] = lum * 0.12           # a touch of red
    out[:, :, 0] = lum * 0.05
    return np.clip(out, 0, 255).astype(np.uint8)


def night_vision():
    st.title("Night Vision Module")
    s1, s2 = st.columns(2)
    max_gain = s1.slider("Sensitivity (max amplification)", 2, 40, 14,
                         help="Higher = brighter in the dark, but noisier.")
    smoothing = s2.slider("Noise removal strength", 2, 24, 12,
                          help="Higher = cleaner, sharper image in still areas. "
                               "Moving objects are kept sharp automatically.")
    st.caption("Software can only amplify the light the webcam actually captures. In a "
               "completely dark room use a dim light, or an IR-capable camera for true night vision.")
    c1, c2 = st.columns(2)
    c1.subheader("Normal")
    c2.subheader("Night Vision")
    run = st.checkbox("Start Night Vision", key="nv_run")
    if not run:
        release_camera()
        st.info("Tick 'Start Night Vision' to start the webcam. Untick it to stop.")
        return
    cap = open_camera()
    if cap is None:
        st.error("Could not open the webcam. Close other apps using it and try again.")
        return
    view = st.empty()                      # ONE image per frame (half the browser traffic)
    state = {}
    pacer = Pacer(15)
    try:
        while True:
            frame = read_frame(cap, mirror=False)
            if frame is None:
                st.error(CAM_HELP)
                break
            nv = apply_night_vision_effect(frame, state, max_gain, smoothing)
            show_image(view, np.hstack((frame, nv)), channels="BGR", output_format="JPEG")
            pacer.wait()
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="MODLS", layout="wide")
    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Object Tracking": object_tracking_page,
        "Cadet Face Registration": cadet_face_registration_page,
        "Cadet Face Recognition": cadet_face_recognition_page,
        "Armory Data Registration": armory_data_registration_page,
        "Armory Data Visualization": armory_data_visualization_page,
        "Intrusion Detection": intrusion_detection_page,
        "Night Vision": night_vision,
    }
    st.sidebar.number_input("Camera index", 0, 5, 0, key="cam_index",
                            help="Change this if the wrong or no camera opens.")
    st.sidebar.selectbox("Camera resolution", ["640x480", "1280x720", "1920x1080"], key="cam_res",
                         help="Capture size requested from the camera. Higher resolution helps "
                              "small or distant objects; the camera may pick the nearest size it supports.")
    st.sidebar.selectbox("Processing width", [640, 960, 1280], key="proc_width",
                         help="Frames are scaled to this width before processing. Larger = more detail "
                              "for tracking, but slower.")
    st.sidebar.checkbox("Infrared / grayscale camera", key="ir_mode",
                        help="Tick when the selected camera index is an IR camera: colour matching "
                             "is replaced by brightness matching and relies more on the recognition model.")
    choice = st.sidebar.radio("Choose a page", list(pages))
    if st.session_state.get("_page") != choice:   # leaving a page turns the camera off
        release_camera()
        st.session_state["_page"] = choice
    pages[choice]()


if __name__ == "__main__":
    main()
