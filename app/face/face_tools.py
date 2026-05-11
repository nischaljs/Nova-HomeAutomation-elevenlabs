import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

FACE_RECO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "face-recognition"
)
sys.path.insert(0, FACE_RECO_PATH)

from face_recognition_system import FaceRecognitionSystem
from face_recognition_system.detector import detect_faces
from face_recognition_system.embedder import generate_embedding
from face_recognition_system.matcher import find_best_match


from app.platform_detect import IS_PI, describe as _platform_describe

DATA_DIR = os.path.join(FACE_RECO_PATH, "face_data")
THRESHOLD = 0.40
DETECT_SCALE = 0.5

# Quality gate thresholds — frames failing any of these are returned as
# {"unknown": True, "low_quality": True} so the presence tracker still
# accumulates time toward publishing face_unknown, but the embedding+match
# path is skipped on bad frames.
# Tuned per platform: Pi camera optics are sharper and tend to fill more
# of the frame, so we can be stricter. Laptop webcams are softer.
if IS_PI:
    MIN_FACE_PX = 60
    MIN_DETECT_SCORE = 0.80
    MIN_BLUR_VAR = 60.0
    MAX_POSE_ASYM = 0.40
else:
    MIN_FACE_PX = 40
    MIN_DETECT_SCORE = 0.70
    MIN_BLUR_VAR = 30.0
    MAX_POSE_ASYM = 0.50

print(f"[FACE] Platform: {_platform_describe()} → "
      f"quality gate (face_px>{MIN_FACE_PX}, score>{MIN_DETECT_SCORE}, "
      f"blur>{MIN_BLUR_VAR}, asym<{MAX_POSE_ASYM})")


def _face_quality_ok(face: dict, image: np.ndarray) -> tuple[bool, str]:
    x, y, w, h = face["bbox"]
    if w < MIN_FACE_PX or h < MIN_FACE_PX:
        return False, f"too_small({w}x{h})"
    if face.get("score", 0.0) < MIN_DETECT_SCORE:
        return False, f"low_score({face.get('score', 0):.2f})"

    x0 = max(0, x)
    y0 = max(0, y)
    crop = image[y0:y0 + h, x0:x0 + w]
    if crop.size == 0:
        return False, "empty_crop"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < MIN_BLUR_VAR:
        return False, f"blurry(var={blur_var:.0f})"

    landmarks = face.get("landmarks")
    if landmarks is not None and len(landmarks) >= 3:
        right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        eye_to_nose_r = abs(float(right_eye[0]) - float(nose[0]))
        eye_to_nose_l = abs(float(left_eye[0]) - float(nose[0]))
        max_dist = max(eye_to_nose_r, eye_to_nose_l)
        if max_dist > 0:
            asym = abs(eye_to_nose_r - eye_to_nose_l) / max_dist
            if asym > MAX_POSE_ASYM:
                return False, f"off_axis(asym={asym:.2f})"

    return True, ""


class FrameBuffer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._frame = None
            cls._instance._timestamp = 0.0
        return cls._instance

    def update(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
            self._timestamp = time.time()

    def get(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._timestamp

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


class RegistrationCollector:
    def __init__(self, max_samples=15, min_gap=1.5):
        self.samples = deque(maxlen=max_samples)
        self.min_gap = min_gap
        self._last_sample = 0.0

    def add(self, frame: np.ndarray, score: float):
        now = time.time()
        if now - self._last_sample < self.min_gap:
            return False
        self.samples.append((frame.copy(), score))
        self._last_sample = now
        return True

    @property
    def count(self):
        return len(self.samples)

    @property
    def full(self):
        return self.count == self.samples.maxlen

    def clear(self):
        self.samples.clear()


class FaceRecognitionBridge:
    def __init__(self, data_dir=DATA_DIR, threshold=THRESHOLD):
        self.system = FaceRecognitionSystem(data_dir=data_dir, threshold=threshold)
        self._recog_lock = threading.Lock()
        # Models load in a background thread so the bridge is usable
        # immediately. recognize_all/register_multi short-circuit while
        # models_ready is False, so SFace can take its 3 minutes to download
        # over slow Pi internet without blocking anything else.
        self._models_ready = threading.Event()
        threading.Thread(
            target=self._validate_models_bg,
            daemon=True,
            name="face-models-init",
        ).start()

    @property
    def models_ready(self) -> bool:
        return self._models_ready.is_set()

    def _validate_models_bg(self):
        try:
            self._validate_models()
            self._models_ready.set()
            print("[FACE] models_ready=True — recognition and registration are live")
        except Exception as e:
            print(f"[FACE] background model validation failed: {type(e).__name__}: {e}")
            print("[FACE] models_ready stays False — recognize/register will short-circuit "
                  "until you restart the app (and ideally fix the network).")

    def _ensure_model(self, name: str, min_size: int):
        """Download `name` model with visible progress reporting if it's
        missing, too small, or has been deleted by a previous validation
        failure. Replaces the silent urlretrieve in face_recognition_system
        with a loop that prints percent + KB/s every ~10%."""
        import urllib.request
        from face_recognition_system.models import MODELS_DIR, MODEL_FILES, MODEL_URLS

        path = MODELS_DIR / MODEL_FILES[name]
        if path.exists():
            size = path.stat().st_size
            if size >= min_size:
                print(f"[FACE] {name}: cached ({size:,} B) ✓")
                return path
            print(f"[FACE] {name}: file on disk is only {size:,} B "
                  f"(expected ≥ {min_size:,} B) — partial download, deleting")
            path.unlink()
        else:
            print(f"[FACE] {name}: not cached yet")

        path.parent.mkdir(parents=True, exist_ok=True)
        url = MODEL_URLS[name]
        print(f"[FACE] {name}: downloading from {url}")

        t0 = time.time()
        last_logged_pct = -10

        def _progress(blocks: int, blocksize: int, totalsize: int):
            nonlocal last_logged_pct
            got = blocks * blocksize
            if totalsize > 0:
                pct = int(min(got, totalsize) * 100 / totalsize)
            else:
                pct = -1
            elapsed = max(time.time() - t0, 0.001)
            kbs = got / elapsed / 1024
            if totalsize > 0 and pct >= last_logged_pct + 10:
                print(f"[FACE] {name}: {pct:3d}%  "
                      f"({got // 1024:>6,} / {totalsize // 1024:,} KB, {kbs:6.0f} KB/s)")
                last_logged_pct = pct

        urllib.request.urlretrieve(url, path, reporthook=_progress)
        final_size = path.stat().st_size
        elapsed = time.time() - t0
        print(f"[FACE] {name}: download complete — {final_size:,} B in {elapsed:.1f}s "
              f"({final_size / elapsed / 1024:.0f} KB/s avg)")
        return path

    def _validate_models(self):
        """Pre-download + verify-load both ONNX models so the first real
        recognize call doesn't pay surprise costs. Logs every step so a
        slow download is visible instead of mysterious silence."""
        from face_recognition_system import detector as det_mod
        from face_recognition_system import embedder as emb_mod

        print("[FACE] validating face recognition models ...")
        self._ensure_model("yunet", min_size=200_000)
        self._ensure_model("sface", min_size=30_000_000)

        checks = [
            ("yunet", lambda: det_mod._get_detector(320, 240), det_mod, "_detector"),
            ("sface", lambda: emb_mod._get_recognizer(), emb_mod, "_recognizer"),
        ]
        for name, load_fn, mod, cache_attr in checks:
            try:
                load_fn()
                print(f"[FACE] {name}: loaded into OpenCV ✓")
            except Exception as e:
                from face_recognition_system.models import MODELS_DIR, MODEL_FILES
                path = MODELS_DIR / MODEL_FILES[name]
                size = path.stat().st_size if path.exists() else 0
                print(f"[FACE] {name}: failed to load (size={size:,} B): "
                      f"{type(e).__name__}: {e}")
                if path.exists():
                    print(f"[FACE] {name}: deleting and retrying download ...")
                    path.unlink()
                setattr(mod, cache_attr, None)
                self._ensure_model(name, min_size=200_000 if name == "yunet" else 30_000_000)
                try:
                    load_fn()
                    print(f"[FACE] {name}: re-downloaded and loaded ✓")
                except Exception as e2:
                    print(f"[FACE] {name}: re-download still failed: {e2}")
        print("[FACE] all models ready")

    def has_face(self, image: np.ndarray) -> bool:
        small = cv2.resize(image, None, fx=DETECT_SCALE, fy=DETECT_SCALE) if image.shape[1] > 320 else image
        faces = detect_faces(small)
        return len(faces) > 0

    def recognize(self, image: np.ndarray) -> dict | None:
        with self._recog_lock:
            try:
                small = cv2.resize(image, None, fx=DETECT_SCALE, fy=DETECT_SCALE) if image.shape[1] > 320 else image
                faces = detect_faces(small)
                if not faces:
                    return None
                face = faces[0]
                ok, why = _face_quality_ok(face, small)
                if not ok:
                    if not getattr(self, "_last_reject", "") == why:
                        print(f"[FACE] quality reject: {why} (suppressing repeats)")
                        self._last_reject = why
                    return {"unknown": True, "face_bbox": face["bbox"], "low_quality": True}
                self._last_reject = ""
                embedding = generate_embedding(small, face["raw"])
                matrix, ids = self.system.storage.load_matrix()
                if matrix is None:
                    return {"unknown": True, "face_bbox": face["bbox"]}
                match = find_best_match(embedding, matrix, ids, THRESHOLD)
                if not match:
                    return {"unknown": True, "face_bbox": face["bbox"]}
                meta = self.system.storage.get_metadata(match["id"]) or {}
                return {
                    "id": match["id"],
                    "name": meta.get("name", "unknown"),
                    "confidence": match["confidence"],
                    "face_bbox": face["bbox"],
                    "unknown": False,
                }
            except Exception as e:
                print(f"[FACE] recognize() exception: {type(e).__name__}: {e}")
                return None

    def recognize_all(self, image: np.ndarray) -> list[dict]:
        """Recognize every face in the frame that passes the quality gate.

        Returns a list of dicts (one per usable face). Each dict carries
        either a known identity (id, name, confidence) or `unknown: True`.
        Low-quality detections are dropped silently so the presence tracker
        doesn't get jitter."""
        if not self._models_ready.is_set():
            return []
        with self._recog_lock:
            try:
                small = (
                    cv2.resize(image, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
                    if image.shape[1] > 320
                    else image
                )
                faces = detect_faces(small)
                if not faces:
                    return []
                matrix, ids = self.system.storage.load_matrix()
                results: list[dict] = []
                for face in faces:
                    ok, why = _face_quality_ok(face, small)
                    if not ok:
                        continue
                    embedding = generate_embedding(small, face["raw"])
                    if matrix is None:
                        results.append({"unknown": True, "face_bbox": face["bbox"]})
                        continue
                    match = find_best_match(embedding, matrix, ids, THRESHOLD)
                    if not match:
                        results.append({"unknown": True, "face_bbox": face["bbox"]})
                        continue
                    meta = self.system.storage.get_metadata(match["id"]) or {}
                    results.append({
                        "id": match["id"],
                        "name": meta.get("name", "unknown"),
                        "confidence": match["confidence"],
                        "face_bbox": face["bbox"],
                        "unknown": False,
                    })
                return results
            except Exception as e:
                print(f"[FACE] recognize_all() exception: {type(e).__name__}: {e}")
                return []

    def register(self, image: np.ndarray, name: str) -> dict | None:
        with self._recog_lock:
            try:
                return self.system.register(image, {"name": name})
            except Exception:
                return None

    def register_multi(self, frames: list[np.ndarray], name: str) -> dict | None:
        if not frames:
            return None
        if not self._models_ready.is_set():
            return None
        with self._recog_lock:
            try:
                embeddings = []
                rejects = []
                for frame in frames:
                    small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE) \
                        if frame.shape[1] > 320 else frame
                    faces = detect_faces(small)
                    if not faces:
                        rejects.append("no_face")
                        continue
                    face = faces[0]
                    ok, why = _face_quality_ok(face, small)
                    if not ok:
                        rejects.append(why)
                        continue
                    emb = generate_embedding(small, face["raw"]).astype(np.float32)
                    n = float(np.linalg.norm(emb))
                    if n <= 0:
                        rejects.append("zero_norm")
                        continue
                    embeddings.append(emb / n)

                print(f"[FACE] register_multi('{name}'): "
                      f"{len(embeddings)} usable / {len(frames)} captured "
                      f"(rejects: {rejects[:5]})")

                if len(embeddings) < 2:
                    return None

                stacked = np.stack(embeddings)
                averaged = stacked.mean(axis=0)
                averaged = averaged / float(np.linalg.norm(averaged))

                identity_id = self.system.storage.save(averaged, {"name": name})
                return {"id": identity_id, "status": "registered", "samples_used": len(embeddings)}
            except Exception as e:
                print(f"[FACE] register_multi failed: {e}")
                return None

    def delete(self, identity_id: str) -> bool:
        return self.system.delete(identity_id)

    def list_identities(self) -> dict:
        return self.system.list_identities()


FACES_BRIDGE = None
REG_COLLECTOR = None
_BRIDGE_INIT_LOCK = threading.Lock()


def get_bridge():
    """Returns the singleton FaceRecognitionBridge.

    Wrapped in a lock so concurrent callers (face pipeline init + a tool
    callback firing while init is still running) don't both try to
    construct their own bridge. The old race caused every tool call to
    spawn a fresh validation that killed the in-progress SFace download.
    """
    global FACES_BRIDGE
    with _BRIDGE_INIT_LOCK:
        if FACES_BRIDGE is None:
            FACES_BRIDGE = FaceRecognitionBridge()
        return FACES_BRIDGE


def get_collector():
    global REG_COLLECTOR
    if REG_COLLECTOR is None:
        REG_COLLECTOR = RegistrationCollector()
    return REG_COLLECTOR
