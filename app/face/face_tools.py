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

# Every detect/recognize/register path in this module resizes the source
# frame to this resolution before handing it to YuNet. Matches what
# VisionPipeline already uses for the live detect loop, so a face that
# looks N pixels wide here looks N pixels wide there — embeddings from
# registration and embeddings from live recognition are produced from
# pixel-equivalent crops, which is required for the cosine matcher to
# behave consistently.
DETECT_TARGET_W = int(os.environ.get("NOVA_DETECT_W", "480"))
DETECT_TARGET_H = int(os.environ.get("NOVA_DETECT_H", "360"))

# Quality gate thresholds — frames failing any of these are returned as
# {"unknown": True, "low_quality": True} so the presence tracker still
# accumulates time toward publishing face_unknown, but the embedding+match
# path is skipped on bad frames.
#
# Loosened for the robot use case (May 2026): the Pi sits on a moving
# platform, not on a fixed kiosk, so we have to recognize people at
# arm's length AND from across the room, with their heads turned
# wherever the conversation is going. The old strict-Pi tuning required
# users to be within ~1 m and looking straight at the camera, which is
# not realistic for a robot wandering past someone in a hallway.
#
# At DETECT_TARGET_W=480 a 40-px face here is ~85 px in a 640-wide
# source — recognizable at ~2–2.5 m with most USB webcams. MAX_POSE_ASYM
# 0.55 allows ~30° of head turn before we reject — about the angle of
# a casual glance.
if IS_PI:
    MIN_FACE_PX = 40
    MIN_DETECT_SCORE = 0.72
    MIN_BLUR_VAR = 40.0
    MAX_POSE_ASYM = 0.55
else:
    MIN_FACE_PX = 35
    MIN_DETECT_SCORE = 0.68
    MIN_BLUR_VAR = 25.0
    MAX_POSE_ASYM = 0.60

# Sun / overexposure handling. A face crop that's mostly blown-out white
# has a very low Laplacian variance not because it's motion-blurred but
# because there's nothing for the kernel to find in a saturated region.
# When the crop's median pixel is above BRIGHT_MEDIAN_PX (≈ 82 % gray)
# we drop the blur floor to BRIGHT_BLUR_VAR_MIN — enough to still reject
# truly featureless frames but not so high that a real outdoor face
# gets thrown away. Reject reason still records the median brightness
# so a 'sun mode' acceptance is distinguishable in logs.
BRIGHT_MEDIAN_PX = 210.0
BRIGHT_BLUR_VAR_MIN = 8.0

# Burst-and-rank registration. With dedup'd capture giving ~25-30 unique
# camera frames in 3.5s, we score every quality-gate-pass frame and only
# embed the top REGISTER_KEEP_TOP — embeddings on a Pi are the expensive
# part (~80 ms each), so doing fewer + better ones is strictly better
# than averaging many noisy ones. Score normalizers are heuristic, not
# ML-trained — they just need to be monotonic (bigger face / sharper /
# more frontal / higher detection confidence = higher score).
REGISTER_KEEP_TOP = int(os.environ.get("NOVA_REGISTER_KEEP_TOP", "12"))
SCORE_FACE_PX_NORM = 120.0   # face width at which size factor saturates to 1
SCORE_BLUR_NORM = 120.0      # Laplacian variance at which blur factor saturates


def _frame_score(face: dict, image: np.ndarray) -> tuple[float, dict]:
    """Score in [0, 1] for ranking quality-gate-passing frames before
    embedding. Returns (score, breakdown) — breakdown is logged so the
    user can see *why* one frame outranked another.
    """
    x, y, w, h = face["bbox"]
    det = float(face.get("score", 0.0))
    size = min(float(w) / SCORE_FACE_PX_NORM, 1.0)

    asym_factor = 1.0
    landmarks = face.get("landmarks")
    if landmarks is not None and len(landmarks) >= 3:
        right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        d_r = abs(float(right_eye[0]) - float(nose[0]))
        d_l = abs(float(left_eye[0]) - float(nose[0]))
        max_d = max(d_r, d_l)
        if max_d > 0:
            asym = abs(d_r - d_l) / max_d
            asym_factor = max(0.0, 1.0 - asym)

    x0, y0 = max(0, x), max(0, y)
    crop = image[y0:y0 + h, x0:x0 + w]
    if crop.size == 0:
        blur_factor = 0.0
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        blur_factor = min(blur_var / SCORE_BLUR_NORM, 1.0)

    score = det * size * asym_factor * blur_factor
    return score, {
        "det": round(det, 2),
        "size": round(size, 2),
        "front": round(asym_factor, 2),
        "sharp": round(blur_factor, 2),
    }

print(f"[FACE] Platform: {_platform_describe()} → "
      f"detect target {DETECT_TARGET_W}x{DETECT_TARGET_H}, "
      f"quality gate (face_px>{MIN_FACE_PX}, score>{MIN_DETECT_SCORE}, "
      f"blur>{MIN_BLUR_VAR} [sun-relaxed to {BRIGHT_BLUR_VAR_MIN} "
      f"when med>{BRIGHT_MEDIAN_PX:.0f}], asym<{MAX_POSE_ASYM})")


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
    # CV_32F instead of CV_64F: half the memory bandwidth on ARM, identical
    # variance to ~6 significant figures — well past what the threshold cares about.
    blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if blur_var < MIN_BLUR_VAR:
        # Sun / overexposure escape hatch — see BRIGHT_MEDIAN_PX comment
        # at the top of the file.
        median_brightness = float(np.median(gray))
        if not (median_brightness >= BRIGHT_MEDIAN_PX
                and blur_var >= BRIGHT_BLUR_VAR_MIN):
            return False, f"blurry(var={blur_var:.0f},med={median_brightness:.0f})"

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


def _resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize an incoming camera frame to (DETECT_TARGET_W, DETECT_TARGET_H)
    so every YuNet call in this module sees the same pixels regardless
    of the source (Picamera2, USB webcam at 640×480, downloaded test
    image). That makes MIN_FACE_PX mean a consistent distance from the
    camera and keeps registration embeddings pixel-equivalent to
    live-recognition embeddings.

    Returns (small, scale_to_source) — the multiplier maps a bbox in
    the small frame back to source coordinates if a caller needs it.
    """
    src_h, src_w = image.shape[:2]
    if src_w == DETECT_TARGET_W and src_h == DETECT_TARGET_H:
        return image, 1.0
    small = cv2.resize(image, (DETECT_TARGET_W, DETECT_TARGET_H))
    return small, src_w / float(DETECT_TARGET_W)


def _pick_primary_face(faces: list[dict]) -> dict | None:
    """In a crowd/family/outdoor scene, the largest-area bbox is the
    person closest to the camera — i.e. the one engaging the robot.
    Taking faces[0] (YuNet's arbitrary first) registers / recognizes the
    wrong person whenever a stranger is incidentally in frame.
    """
    if not faces:
        return None
    return max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])


def _crop_face_with_margin(image: np.ndarray, bbox, margin_frac: float = 0.25,
                           thumb_px: int = 256) -> np.ndarray | None:
    """Crop a face region with a percentage margin and resize to a square
    thumbnail. Margin gives the /setup UI grid a bit of context around
    the face instead of a tight head-shot. Must match the cropping that
    the /setup upload path uses (routes.py::_crop_face_to_thumb) so
    register-time thumbnails and upload-time thumbnails look the same."""
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    margin = int(margin_frac * max(w, h))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    try:
        return cv2.resize(crop, (thumb_px, thumb_px), interpolation=cv2.INTER_AREA)
    except cv2.error:
        return None


def _save_per_photo_artifacts(
    storage,
    identity_id: str,
    usable: list[tuple[np.ndarray, dict, np.ndarray]],
) -> int:
    """Persist one JPEG thumbnail + one per-photo .npy embedding under
    face_data/photos/<identity_id>/ for every usable frame from a
    successful registration. The /setup UI reads these to render
    thumbnails and to re-average the identity when the user adds or
    removes photos. Best-effort — registration succeeds even if this
    fails. Returns the count actually written.

    Write order is .npy first, then .jpg. The /setup `_list_photo_ids`
    filter requires both files to exist, so a crash between the two
    leaves an orphan .npy that's invisible to listing AND to re-
    averaging (re-averaging also goes through _list_photo_ids). The
    opposite order would risk a future re-average implementation
    picking up an orphan .npy. Same rule as routes.py::_save_photo_files.
    """
    if not usable:
        return 0
    import uuid
    photos_dir = storage.embeddings_dir.parent / "photos" / identity_id
    try:
        photos_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[FACE] photos: mkdir failed for {photos_dir}: {e} (skipping)")
        return 0
    written = 0
    for small, face, emb in usable:
        thumb = _crop_face_with_margin(small, face["bbox"])
        if thumb is None:
            continue
        photo_id = uuid.uuid4().hex[:12]
        npy_path = photos_dir / f"{photo_id}.npy"
        jpg_path = photos_dir / f"{photo_id}.jpg"
        try:
            np.save(npy_path, emb.astype(np.float32))
            ok = cv2.imwrite(str(jpg_path), thumb,
                             [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                # imwrite failed silently — clean up the orphan .npy so
                # re-averaging stays consistent with what _list_photo_ids
                # advertises.
                npy_path.unlink(missing_ok=True)
                continue
            written += 1
        except OSError as e:
            print(f"[FACE] photos: write failed for {photo_id}: {e} (skipping)")
            # Best-effort cleanup of any partial file
            npy_path.unlink(missing_ok=True)
            jpg_path.unlink(missing_ok=True)
    return written


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
        # In-memory cache of the embeddings matrix. Without this, every
        # recognize call reads the JSON from disk inside the locked
        # critical section — small but real (~5-10 ms on a Pi 4) and
        # unnecessary because the matrix only changes when we
        # register or delete an identity.
        self._matrix_cache: tuple | None = None
        self._matrix_cache_lock = threading.Lock()
        threading.Thread(
            target=self._validate_models_bg,
            daemon=True,
            name="face-models-init",
        ).start()

    def _load_matrix_cached(self):
        """Return (matrix, ids), reading from disk only the first time
        after a register/delete invalidation. Thread-safe — concurrent
        callers will all see the same cached tuple."""
        with self._matrix_cache_lock:
            if self._matrix_cache is None:
                self._matrix_cache = self.system.storage.load_matrix()
            return self._matrix_cache

    def _invalidate_matrix_cache(self):
        with self._matrix_cache_lock:
            self._matrix_cache = None

    @property
    def models_ready(self) -> bool:
        return self._models_ready.is_set()

    def wait_models_ready(self, timeout: float | None = None) -> bool:
        """Block until the background validation thread sets models_ready.
        Returns True if ready, False if timeout elapsed first."""
        return self._models_ready.wait(timeout)

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
        failure. Logs every 5% plus an ETA so a 3-minute SFace download
        on slow Pi internet is never mistaken for a hang."""
        import urllib.request
        from face_recognition_system.models import MODELS_DIR, MODEL_FILES, MODEL_URLS

        path = MODELS_DIR / MODEL_FILES[name]
        if path.exists():
            size = path.stat().st_size
            if size >= min_size:
                print(f"[FACE] {name}: cached ({size:,} B) ✓ — instant load")
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
        last_logged_pct = -5
        last_heartbeat = t0

        def _progress(blocks: int, blocksize: int, totalsize: int):
            nonlocal last_logged_pct, last_heartbeat
            got = blocks * blocksize
            now = time.time()
            elapsed = max(now - t0, 0.001)
            kbs = got / elapsed / 1024

            if totalsize > 0:
                pct = int(min(got, totalsize) * 100 / totalsize)
                remaining_kb = max(0, (totalsize - got) / 1024)
                eta_s = remaining_kb / max(kbs, 0.001)
                if pct >= last_logged_pct + 5:
                    print(f"[FACE] {name}: {pct:3d}%  "
                          f"({got // 1024:>6,} / {totalsize // 1024:,} KB, "
                          f"{kbs:6.0f} KB/s, ETA {eta_s:5.0f}s)")
                    last_logged_pct = pct
                    last_heartbeat = now
                elif now - last_heartbeat >= 15:
                    print(f"[FACE] {name}: still downloading… "
                          f"{got // 1024:,} KB at {kbs:.0f} KB/s")
                    last_heartbeat = now

        urllib.request.urlretrieve(url, path, reporthook=_progress)
        final_size = path.stat().st_size
        elapsed = time.time() - t0
        print(f"[FACE] {name}: ✓ download complete — {final_size:,} B in {elapsed:.1f}s "
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
        small, _ = _resize_for_detection(image)
        faces = detect_faces(small)
        return len(faces) > 0

    def recognize(self, image: np.ndarray) -> dict | None:
        with self._recog_lock:
            try:
                small, _ = _resize_for_detection(image)
                faces = detect_faces(small)
                if not faces:
                    return None
                # Pick the largest face — closest to camera, most likely
                # the engaged user (see _pick_primary_face).
                face = _pick_primary_face(faces)
                ok, why = _face_quality_ok(face, small)
                if not ok:
                    if not getattr(self, "_last_reject", "") == why:
                        print(f"[FACE] quality reject: {why} (suppressing repeats)")
                        self._last_reject = why
                    return {"unknown": True, "face_bbox": face["bbox"], "low_quality": True}
                self._last_reject = ""
                embedding = generate_embedding(small, face["raw"])
                matrix, ids = self._load_matrix_cached()
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
                small, _ = _resize_for_detection(image)
                faces = detect_faces(small)
                if not faces:
                    return []
                return self._recognize_in_locked(small, faces)
            except Exception as e:
                print(f"[FACE] recognize_all() exception: {type(e).__name__}: {e}")
                return []

    def recognize_in(self, image: np.ndarray, faces: list[dict]) -> list[dict]:
        """Recognize a *pre-detected* set of faces.

        Used by VisionPipeline so we don't pay for detection twice (once
        for the live preview at 6 Hz, again for recognition at 2 Hz).
        The caller is responsible for passing the same `image` that the
        bboxes were produced from — otherwise SFace will embed the wrong
        crop and we'll get garbage matches."""
        if not self._models_ready.is_set() or not faces:
            return []
        with self._recog_lock:
            try:
                return self._recognize_in_locked(image, faces)
            except Exception as e:
                print(f"[FACE] recognize_in() exception: {type(e).__name__}: {e}")
                return []

    def _recognize_in_locked(self, image: np.ndarray, faces: list[dict]) -> list[dict]:
        matrix, ids = self._load_matrix_cached()
        results: list[dict] = []
        for face in faces:
            ok, why = _face_quality_ok(face, image)
            if not ok:
                continue
            embedding = generate_embedding(image, face["raw"])
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

    def register(self, image: np.ndarray, name: str) -> dict | None:
        with self._recog_lock:
            try:
                result = self.system.register(image, {"name": name})
                self._invalidate_matrix_cache()
                return result
            except Exception:
                return None

    def register_multi(self, frames: list[np.ndarray], name: str) -> dict:
        """Average N face embeddings into one identity.

        Always returns a dict with at least `ok` and `rejects` so the
        caller can produce a useful error message — the old "return None"
        path lost all diagnostic info and made the agent's failure reply
        generic ("I couldn't get a clear look at you").

        Returns:
          {"ok": True, "id": str, "samples_used": int, "rejects": list[str]}  on success
          {"ok": False, "samples_used": int, "rejects": list[str]}             on failure
        """
        if not frames:
            return {"ok": False, "samples_used": 0, "rejects": ["no_frames"]}
        if not self._models_ready.is_set():
            return {"ok": False, "samples_used": 0, "rejects": ["models_not_ready"]}
        with self._recog_lock:
            try:
                rejects: list[str] = []
                # Pass 1: detect + quality gate + score every input
                # frame. Embedding is the expensive part on a Pi (~80 ms
                # per call vs ~5 ms for detect), so we defer it until
                # we know which frames are worth embedding. Each
                # candidate carries the data the embed pass needs
                # (the small frame and the raw YuNet output) plus the
                # score we'll rank by.
                candidates: list[tuple[float, np.ndarray, dict, dict]] = []
                for frame in frames:
                    small, _ = _resize_for_detection(frame)
                    faces = detect_faces(small)
                    if not faces:
                        rejects.append("no_face")
                        continue
                    face = _pick_primary_face(faces)
                    ok, why = _face_quality_ok(face, small)
                    if not ok:
                        rejects.append(why)
                        continue
                    score, breakdown = _frame_score(face, small)
                    candidates.append((score, small, face, breakdown))

                # Pass 2: rank, take top REGISTER_KEEP_TOP, embed only
                # those. Keeps memory bounded (we don't hold all
                # candidate frames *and* their embeddings) and burns
                # the SFace budget on the best samples instead of
                # averaging in mediocre ones.
                candidates.sort(key=lambda c: c[0], reverse=True)
                top = candidates[:REGISTER_KEEP_TOP]

                embeddings: list[np.ndarray] = []
                usable: list[tuple[np.ndarray, dict, np.ndarray]] = []
                for score, small, face, _bk in top:
                    emb = generate_embedding(small, face["raw"]).astype(np.float32)
                    n = float(np.linalg.norm(emb))
                    if n <= 0:
                        rejects.append("zero_norm")
                        continue
                    emb_norm = emb / n
                    embeddings.append(emb_norm)
                    usable.append((small, face, emb_norm))

                # Histogram of reject *reasons* (not the noisy per-frame
                # detail) is what tells you at a glance what's wrong —
                # `{'too_small': 5, 'blurry': 1}` vs scrolling 6 strings.
                reject_summary: dict[str, int] = {}
                for r in rejects:
                    key = r.split("(", 1)[0]
                    reject_summary[key] = reject_summary.get(key, 0) + 1

                top_scores = [round(c[0], 2) for c in top]
                print(f"[FACE] register_multi('{name}'): "
                      f"captured={len(frames)} passed_gate={len(candidates)} "
                      f"embedded={len(embeddings)} (top {len(top)} scored "
                      f"{top_scores})  reasons={reject_summary or '{}'}")

                if len(embeddings) < 2:
                    return {"ok": False, "samples_used": len(embeddings), "rejects": rejects}

                stacked = np.stack(embeddings)
                averaged = stacked.mean(axis=0)
                averaged = averaged / float(np.linalg.norm(averaged))

                identity_id = self.system.storage.save(averaged, {"name": name})
                self._invalidate_matrix_cache()
                written = _save_per_photo_artifacts(
                    self.system.storage, identity_id, usable
                )
                if written:
                    print(f"[FACE] register_multi('{name}'): "
                          f"saved {written} reference photo(s) for /setup UI")
                return {
                    "ok": True,
                    "id": identity_id,
                    "samples_used": len(embeddings),
                    "rejects": rejects,
                }
            except Exception as e:
                print(f"[FACE] register_multi failed: {e}")
                return {
                    "ok": False,
                    "samples_used": 0,
                    "rejects": [f"exception:{type(e).__name__}"],
                }

    def delete(self, identity_id: str) -> bool:
        result = self.system.delete(identity_id)
        if result:
            self._invalidate_matrix_cache()
        return result

    def reinforce_by_name(
        self,
        image: np.ndarray,
        name: str,
        new_weight: float = 0.30,
    ) -> dict:
        """Online learning: when an uncertain match is confirmed by the
        user, blend the current frame's embedding INTO the existing
        identity's stored embedding. Over weeks of use, an identity's
        average embedding tracks how the user actually looks now
        (different lighting, hairstyle, weight, glasses) instead of
        being frozen at registration time.

        `new_weight` is how much the new frame counts vs the existing
        average — 0.30 means 70 % old + 30 % new, which lets a single
        confirmation nudge the average without one bad frame ruining
        the identity. Lower = more conservative, higher = more
        responsive to recent appearance.

        Returns: {"ok": bool, "reason": str|None, "id": str|None}.
        Failures fall through with a reason the agent can read aloud
        ("you're not in frame right now", "I'm not sure which face is
        yours", etc.).
        """
        if not self._models_ready.is_set():
            return {"ok": False, "reason": "models_not_ready", "id": None}
        if not name:
            return {"ok": False, "reason": "no_name", "id": None}

        # Find the identity_id by name. case-insensitive prefix so
        # 'Nischal' matches 'nischal' / 'Nischal Bhattarai'.
        target_name = name.strip().lower()
        identity_id = None
        identities = self.system.storage.list_identities() or {}
        for fid, meta in identities.items():
            stored = (meta or {}).get("name", "").strip().lower()
            if stored == target_name:
                identity_id = fid
                break
            if not identity_id and stored.startswith(target_name):
                identity_id = fid  # fallback to prefix match
        if identity_id is None:
            return {"ok": False, "reason": "name_not_in_db", "id": None}

        with self._recog_lock:
            try:
                # Detect the largest face in the current frame so we
                # know we have a single, clear subject to reinforce
                # against. Same detect target the rest of the pipeline
                # uses so coordinates line up.
                small, _ = _resize_for_detection(image)
                faces = detect_faces(small) or []
                if not faces:
                    return {"ok": False, "reason": "no_face_in_frame", "id": identity_id}
                face = _pick_primary_face(faces)
                ok, why = _face_quality_ok(face, small)
                if not ok:
                    return {"ok": False, "reason": f"quality:{why}", "id": identity_id}

                new_emb = generate_embedding(small, face["raw"]).astype(np.float32)
                n = float(np.linalg.norm(new_emb))
                if n <= 0:
                    return {"ok": False, "reason": "zero_norm_embedding", "id": identity_id}
                new_emb = new_emb / n

                # Load the existing stored embedding (single .npy per
                # identity, already L2-normalized from register_multi).
                from face_recognition_system.models import MODELS_DIR  # noqa: F401
                existing_emb = None
                for emb, fid in zip(*self.system.storage.load_all()):
                    if fid == identity_id:
                        existing_emb = emb.astype(np.float32)
                        break
                if existing_emb is None:
                    return {"ok": False, "reason": "embedding_missing", "id": identity_id}

                # Weighted average, then renormalize to keep all
                # embeddings on the same hypersphere the matcher
                # assumes. A naive average without renormalize would
                # bias future cosine similarities.
                blended = (1.0 - new_weight) * existing_emb + new_weight * new_emb
                blended = blended / float(np.linalg.norm(blended))

                # Persist by writing back to the embedding file
                # directly. storage.save() would create a *new* row
                # with a fresh UUID, which is not what we want — we
                # want to update the existing identity in place.
                emb_path = (
                    self.system.storage.embeddings_dir
                    / f"{identity_id}.npy"
                )
                np.save(emb_path, blended)
                # Invalidate both caches: the bridge's matrix cache,
                # and FaceStorage's internal cache (its load_all keeps
                # a snapshot we just wrote past).
                self._invalidate_matrix_cache()
                try:
                    self.system.storage._cache_valid = False
                    self.system.storage._matrix = None
                except AttributeError:
                    pass
                meta = identities.get(identity_id) or {}
                print(f"[FACE] reinforced '{meta.get('name', name)}' "
                      f"(id={identity_id}) with new frame at "
                      f"weight={new_weight}")
                return {"ok": True, "reason": None, "id": identity_id}
            except Exception as e:
                print(f"[FACE] reinforce_by_name failed: "
                      f"{type(e).__name__}: {e}")
                return {"ok": False, "reason": f"exception:{type(e).__name__}", "id": identity_id}

    def list_identities(self) -> dict:
        return self.system.list_identities()

    def invalidate_cache_public(self) -> None:
        """External callers (e.g. the /setup admin UI) use this after they
        mutate embeddings or metadata on disk so the next recognize() call
        re-reads from disk instead of using a stale matrix cache. Also
        clears FaceStorage's internal cache so load_all/load_matrix
        actually re-scan."""
        self._invalidate_matrix_cache()
        try:
            self.system.storage._cache_valid = False
            self.system.storage._matrix = None
        except AttributeError:
            pass

    def detect_contamination(
        self,
        identity_id: str,
        z_threshold: float = 2.0,
    ) -> list[str]:
        """Flag per-photo embeddings that sit far from this identity's
        centroid — i.e. probably a different person that got into the
        record by accident (someone walked into a multi-person frame
        during register, or a mis-clicked upload in /setup).

        Returns a list of photo_id strings (.npy/.jpg filename stems
        under face_data/photos/<id>/) that look like outliers. Empty
        list when the identity is internally consistent or has too few
        photos to judge.

        Used by the /setup UI to surface a 'review these photos' badge
        next to suspect identities. Not used by the live recognition
        path — it's a curation/cleanup tool only.
        """
        photos_dir = (
            self.system.storage.embeddings_dir.parent
            / "photos"
            / identity_id
        )
        if not photos_dir.exists():
            return []
        embs: list[np.ndarray] = []
        ids: list[str] = []
        for p in sorted(photos_dir.glob("*.npy")):
            try:
                e = np.load(p).astype(np.float32)
            except (OSError, ValueError):
                continue
            n = float(np.linalg.norm(e))
            if n <= 0:
                continue
            embs.append(e / n)
            ids.append(p.stem)
        # Need a few samples to have any meaningful distribution. With
        # 3, one outlier pulls the mean too much to detect itself.
        if len(embs) < 4:
            return []
        M = np.stack(embs)
        centroid = M.mean(axis=0)
        centroid = centroid / float(np.linalg.norm(centroid))
        sims = M @ centroid
        mean = float(np.mean(sims))
        std = float(np.std(sims))
        if std < 1e-4:
            return []
        return [
            ids[i]
            for i, s in enumerate(sims)
            if (mean - float(s)) > z_threshold * std
        ]


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
