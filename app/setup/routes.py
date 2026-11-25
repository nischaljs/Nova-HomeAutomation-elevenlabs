"""FastAPI router for the /setup admin UI.

Routes:
    GET    /setup                                            HTML page
    GET    /setup/api/identities                             list identities (JSON)
    POST   /setup/api/identities/{id}/rename                 update name
    DELETE /setup/api/identities/{id}                        remove identity
    GET    /setup/api/identities/{id}/photos                 list photos
    POST   /setup/api/identities/{id}/photos                 upload one or more
    GET    /setup/api/identities/{id}/photos/{photo_id}      serve a JPG
    DELETE /setup/api/identities/{id}/photos/{photo_id}      remove one photo

Storage layout it writes to (alongside the existing embeddings/ dir):

    face_data/
        embeddings/<id>.npy          # averaged embedding (live recognizer)
        metadata.json                # {id: {name, ...}}
        photos/<id>/
            <photo_uuid>.jpg         # cropped face crop, ~256 px
            <photo_uuid>.npy         # per-photo embedding (used for re-averaging)

Re-averaging mirrors what register_multi does in face_tools.py:
    stacked = np.stack(per_photo_embeddings)
    averaged = stacked.mean(axis=0)
    averaged /= np.linalg.norm(averaged)

After any mutating op (rename, delete, upload, photo-delete) we call
`bridge.invalidate_cache_public()` so the next recognize() picks up the
new state.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
)

from app.face.face_tools import (
    DATA_DIR,
    DETECT_SCALE,
    FaceRecognitionBridge,
    _face_quality_ok,
    get_bridge,
)
from face_recognition_system.detector import detect_faces
from face_recognition_system.embedder import generate_embedding


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHOTOS_DIR = Path(DATA_DIR) / "photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

THUMB_PX = 256
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILES_PER_REQUEST = 10
MIN_PHOTOS_AFTER_DELETE = 2  # mirrors register_multi's `len(embeddings) < 2` guard

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

TEMPLATE_DIR = Path(__file__).parent / "templates"


router = APIRouter(prefix="/setup", tags=["setup"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_photo_dir(identity_id: str) -> Path:
    return PHOTOS_DIR / identity_id


def _list_photo_ids(identity_id: str) -> list[str]:
    """Return all photo_ids for an identity, sorted for stable thumbnails.

    A photo is "present" iff both <id>.jpg AND <id>.npy exist — anything
    half-written from a crash gets ignored rather than corrupting the
    averaged embedding.
    """
    pdir = _identity_photo_dir(identity_id)
    if not pdir.exists():
        return []
    ids: list[str] = []
    for jpg in pdir.glob("*.jpg"):
        stem = jpg.stem
        if (pdir / f"{stem}.npy").exists():
            ids.append(stem)
    ids.sort()
    return ids


def _largest_photo_id(identity_id: str) -> str | None:
    """Pick a thumbnail. All crops are written at the same size so we
    just take the lexicographically-first one — deterministic and free."""
    ids = _list_photo_ids(identity_id)
    return ids[0] if ids else None


def _bridge() -> FaceRecognitionBridge:
    return get_bridge()


def _identity_or_404(identity_id: str) -> dict:
    meta = _bridge().system.storage.get_metadata(identity_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"identity {identity_id} not found")
    return meta


def _decode_upload_to_bgr(raw: bytes) -> np.ndarray | None:
    """Decode an image upload into a BGR numpy array, or return None if
    OpenCV can't parse it. We don't trust the content-type header to mean
    the file is actually a valid image."""
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


def _crop_face_to_thumb(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the face region with a small margin and resize to THUMB_PX."""
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    # 25 % margin around the face — gives context but not too much background.
    margin = int(0.25 * max(w, h))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None  # type: ignore[return-value]
    return cv2.resize(crop, (THUMB_PX, THUMB_PX), interpolation=cv2.INTER_AREA)


def _scale_factor_for(image: np.ndarray) -> float:
    """Mirror the bridge's DETECT_SCALE behaviour so coordinates line up
    with what _face_quality_ok expects."""
    return DETECT_SCALE if image.shape[1] > 320 else 1.0


def _detect_primary_face(image: np.ndarray) -> tuple[dict | None, np.ndarray, str | None]:
    """Detect the largest face and run the same quality gate the bridge
    uses. Returns (face, small_image, reject_reason).

    On success: face != None, reject_reason is None.
    On reject:  face is None, reject_reason explains why.
    """
    scale = _scale_factor_for(image)
    small = (
        cv2.resize(image, None, fx=scale, fy=scale)
        if scale != 1.0
        else image
    )
    faces = detect_faces(small)
    if not faces:
        return None, small, "no_face"
    face = faces[0]
    ok, why = _face_quality_ok(face, small)
    if not ok:
        return None, small, why
    return face, small, None


def _save_photo_files(
    identity_id: str,
    photo_id: str,
    small_image: np.ndarray,
    face: dict,
    embedding: np.ndarray,
) -> None:
    """Write <photo_id>.jpg and <photo_id>.npy atomically-ish.

    We write the .npy first so a crash mid-write leaves the .jpg
    orphaned (which _list_photo_ids ignores) rather than the .npy
    orphaned (which would pollute re-averaging)."""
    pdir = _identity_photo_dir(identity_id)
    pdir.mkdir(parents=True, exist_ok=True)

    crop = _crop_face_to_thumb(small_image, face["bbox"])
    if crop is None:
        raise HTTPException(status_code=500, detail="empty_crop")

    npy_path = pdir / f"{photo_id}.npy"
    jpg_path = pdir / f"{photo_id}.jpg"

    np.save(npy_path, embedding.astype(np.float32))
    ok = cv2.imwrite(str(jpg_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        npy_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="failed_to_write_jpg")


def _reaverage_and_write(identity_id: str) -> int:
    """Load every per-photo .npy for this identity, average + L2-normalize,
    and overwrite the live averaged embedding at embeddings/<id>.npy.

    Returns the number of photos that contributed. Mirrors the math at
    face_tools.py:452-454.
    """
    pdir = _identity_photo_dir(identity_id)
    embs: list[np.ndarray] = []
    for pid in _list_photo_ids(identity_id):
        try:
            e = np.load(pdir / f"{pid}.npy").astype(np.float32)
        except Exception:
            continue
        n = float(np.linalg.norm(e))
        if n <= 0:
            continue
        embs.append(e / n)

    if not embs:
        return 0

    stacked = np.stack(embs)
    averaged = stacked.mean(axis=0)
    averaged = averaged / float(np.linalg.norm(averaged))

    storage = _bridge().system.storage
    out_path = storage.embeddings_dir / f"{identity_id}.npy"
    np.save(out_path, averaged.astype(np.float32))
    return len(embs)


def _photo_url(identity_id: str, photo_id: str) -> str:
    return f"/setup/api/identities/{identity_id}/photos/{photo_id}"


def _identity_summary(identity_id: str, meta: dict) -> dict:
    pids = _list_photo_ids(identity_id)
    thumb = pids[0] if pids else None
    return {
        "id": identity_id,
        "name": meta.get("name", ""),
        "photo_count": len(pids),
        "thumb_url": _photo_url(identity_id, thumb) if thumb else None,
    }


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def setup_index() -> HTMLResponse:
    """Serve the static admin page. All data is loaded client-side via
    /setup/api/identities so the HTML itself is just shell + vanilla JS."""
    html_path = TEMPLATE_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@router.get("/api/identities")
async def list_identities() -> JSONResponse:
    bridge = _bridge()
    identities = bridge.system.storage.list_identities() or {}
    out = [_identity_summary(iid, meta or {}) for iid, meta in identities.items()]
    # stable-ish ordering by name so the UI doesn't shuffle on every refresh
    out.sort(key=lambda r: (r["name"].lower(), r["id"]))
    return JSONResponse(out)


@router.get("/api/identities/{identity_id}")
async def get_identity(identity_id: str) -> JSONResponse:
    meta = _identity_or_404(identity_id)
    summary = _identity_summary(identity_id, meta)
    pids = _list_photo_ids(identity_id)
    summary["photos"] = [
        {"id": pid, "url": _photo_url(identity_id, pid)} for pid in pids
    ]
    return JSONResponse(summary)


@router.post("/api/identities/{identity_id}/rename")
async def rename_identity(identity_id: str, payload: dict) -> JSONResponse:
    _identity_or_404(identity_id)
    new_name = (payload or {}).get("name", "")
    if not isinstance(new_name, str):
        raise HTTPException(status_code=400, detail="name must be a string")
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="name cannot be empty")
    if len(new_name) > 200:
        raise HTTPException(status_code=400, detail="name too long")

    storage = _bridge().system.storage
    meta = dict(storage._metadata.get(identity_id) or {})
    meta["name"] = new_name
    storage._metadata[identity_id] = meta
    storage._write_metadata()

    # Metadata only — matrix cache doesn't need invalidating for a rename,
    # but the bridge's cached metadata-by-id lookup might. Cheap to call.
    _bridge().invalidate_cache_public()
    return JSONResponse({"ok": True, "id": identity_id, "name": new_name})


@router.delete("/api/identities/{identity_id}")
async def delete_identity(identity_id: str) -> JSONResponse:
    _identity_or_404(identity_id)
    bridge = _bridge()
    # storage.delete() handles metadata + embeddings/<id>.npy
    ok = bridge.system.storage.delete(identity_id)
    if not ok:
        raise HTTPException(status_code=404, detail="identity disappeared")
    # photos/<id>/ is ours to clean up
    pdir = _identity_photo_dir(identity_id)
    if pdir.exists():
        shutil.rmtree(pdir, ignore_errors=True)
    bridge.invalidate_cache_public()
    return JSONResponse({"ok": True, "id": identity_id})


@router.get("/api/identities/{identity_id}/photos")
async def list_photos(identity_id: str) -> JSONResponse:
    _identity_or_404(identity_id)
    pids = _list_photo_ids(identity_id)
    return JSONResponse({
        "id": identity_id,
        "photos": [
            {"id": pid, "url": _photo_url(identity_id, pid)} for pid in pids
        ],
    })


@router.post("/api/identities/{identity_id}/photos")
async def upload_photos(
    identity_id: str,
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    _identity_or_404(identity_id)

    if not files:
        raise HTTPException(status_code=400, detail="no files")
    # Truncate rather than reject — the UI can split a big upload but it's
    # nicer if a slightly-over-limit batch still succeeds for the first N.
    files = files[:MAX_FILES_PER_REQUEST]

    results: list[dict] = []
    accepted_any = False
    bridge = _bridge()

    if not bridge.models_ready:
        raise HTTPException(
            status_code=503,
            detail="face models not ready yet — wait a moment and retry",
        )

    for uf in files:
        result = {"filename": uf.filename or "(unnamed)", "accepted": False}
        try:
            ctype = (uf.content_type or "").lower()
            if ctype and ctype not in ALLOWED_CONTENT_TYPES:
                result["reason"] = f"bad_content_type:{ctype}"
                results.append(result)
                continue

            raw = await uf.read()
            if len(raw) == 0:
                result["reason"] = "empty"
                results.append(result)
                continue
            if len(raw) > MAX_FILE_BYTES:
                result["reason"] = f"too_large({len(raw)}B>{MAX_FILE_BYTES}B)"
                results.append(result)
                continue

            img = _decode_upload_to_bgr(raw)
            if img is None:
                result["reason"] = "not_a_decodable_image"
                results.append(result)
                continue

            face, small, why = _detect_primary_face(img)
            if face is None:
                result["reason"] = f"quality:{why}"
                results.append(result)
                continue

            embedding = generate_embedding(small, face["raw"]).astype(np.float32)
            n = float(np.linalg.norm(embedding))
            if n <= 0:
                result["reason"] = "zero_norm"
                results.append(result)
                continue
            embedding /= n

            photo_id = uuid.uuid4().hex[:12]
            _save_photo_files(identity_id, photo_id, small, face, embedding)
            result["accepted"] = True
            result["photo_id"] = photo_id
            result["url"] = _photo_url(identity_id, photo_id)
            accepted_any = True
        except HTTPException:
            raise
        except Exception as e:
            result["reason"] = f"exception:{type(e).__name__}:{e}"
        finally:
            try:
                await uf.close()
            except Exception:
                pass
        results.append(result)

    new_count = 0
    if accepted_any:
        new_count = _reaverage_and_write(identity_id)
        bridge.invalidate_cache_public()

    return JSONResponse({
        "id": identity_id,
        "results": results,
        "accepted_count": sum(1 for r in results if r["accepted"]),
        "total_photos": len(_list_photo_ids(identity_id)),
        "averaged_from": new_count,
    })


@router.get("/api/identities/{identity_id}/photos/{photo_id}")
async def get_photo(identity_id: str, photo_id: str) -> FileResponse:
    _identity_or_404(identity_id)
    # Defense against path traversal — photo_id comes from URL.
    if "/" in photo_id or ".." in photo_id or "\\" in photo_id:
        raise HTTPException(status_code=400, detail="bad photo_id")
    jpg = _identity_photo_dir(identity_id) / f"{photo_id}.jpg"
    if not jpg.exists():
        raise HTTPException(status_code=404, detail="photo not found")
    return FileResponse(str(jpg), media_type="image/jpeg")


@router.delete("/api/identities/{identity_id}/photos/{photo_id}")
async def delete_photo(identity_id: str, photo_id: str) -> JSONResponse:
    _identity_or_404(identity_id)
    if "/" in photo_id or ".." in photo_id or "\\" in photo_id:
        raise HTTPException(status_code=400, detail="bad photo_id")

    pdir = _identity_photo_dir(identity_id)
    jpg = pdir / f"{photo_id}.jpg"
    npy = pdir / f"{photo_id}.npy"
    if not jpg.exists() and not npy.exists():
        raise HTTPException(status_code=404, detail="photo not found")

    # Guard: same rule register_multi enforces — we need at least 2 photos
    # to keep an identity stable. Deleting below that would either leave a
    # single-frame embedding (brittle) or zero embeddings (broken match).
    remaining_after = len(_list_photo_ids(identity_id)) - 1
    if remaining_after < MIN_PHOTOS_AFTER_DELETE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"refusing to delete: would leave {remaining_after} photo(s), "
                f"minimum is {MIN_PHOTOS_AFTER_DELETE}. "
                "Upload more photos first, or delete the whole identity."
            ),
        )

    jpg.unlink(missing_ok=True)
    npy.unlink(missing_ok=True)
    count = _reaverage_and_write(identity_id)
    _bridge().invalidate_cache_public()
    return JSONResponse({
        "ok": True,
        "id": identity_id,
        "photo_id": photo_id,
        "total_photos": len(_list_photo_ids(identity_id)),
        "averaged_from": count,
    })
