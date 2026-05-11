import sounddevice as sd
import numpy as np
import scipy.signal
from openwakeword.model import Model

MIC_RATE = 44100
DEVICE_INDEX = 1

# -----------------------------
# LOAD MODEL
# -----------------------------
model = Model()  # loads default wake word models

print("👂 OpenWakeWord system started...")

# -----------------------------
# CALLBACK
# -----------------------------
def callback(indata, frames, time, status):
    if status:
        return

    audio = indata[:, 0].astype(np.float32)

    # 🔁 resample to 16kHz
    audio_16k = scipy.signal.resample_poly(
        audio,
        16000,
        MIC_RATE
    ).astype(np.float32)

    # -----------------------------
    # RUN DETECTION
    # -----------------------------
    prediction = model.predict(audio_16k)

    # print scores for debugging
    for key, score in prediction.items():
        print(f"{key}: {score:.3f}")

        # trigger condition
        if score > 0.6:
            print("🔥 WAKE WORD DETECTED!")

# -----------------------------
# STREAM
# -----------------------------
with sd.InputStream(
    device=DEVICE_INDEX,
    samplerate=MIC_RATE,
    channels=1,
    dtype="int16",
    blocksize=1024,
    callback=callback
):
    while True:
        pass