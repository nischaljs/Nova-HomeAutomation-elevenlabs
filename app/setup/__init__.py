"""Admin UI for managing saved face identities.

Mounts a small server-rendered HTML page at /setup plus a JSON API at
/setup/api/* for renaming, deleting, and uploading photos for any
registered face.

The package is deliberately self-contained: it talks to the existing
FaceRecognitionBridge via its public API and reads/writes the storage
layout directly under face_data/photos/ — no edits to the bridge
internals required (one tiny public cache-invalidation helper was the
only addition).
"""

from .routes import router  # noqa: F401
