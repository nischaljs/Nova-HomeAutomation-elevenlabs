# Running Nova on the Raspberry Pi (`elevenlabs` branch)

This branch is **Pi-only**. No ESP32. The Pi handles the camera, the mic, the speaker, and the network call to ElevenLabs.

## Hardware

- Raspberry Pi 4 (4 GB or 8 GB) — anything older will be tight on the face-recognition model.
- USB webcam (any UVC camera works — Logitech C270, C920, etc.).
- USB microphone (a USB headset mic or a USB lavalier is fine).
- 3.5 mm speaker plugged into the Pi's audio jack. Powered desktop speakers are loudest.
- SD card (32 GB minimum, 64 GB recommended).
- Reliable internet — ElevenLabs streams over WebSocket, so a flaky Wi-Fi link will introduce stutter.

## One-time system setup

```bash
sudo apt update
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    build-essential \
    portaudio19-dev \
    libopencv-dev \
    libatlas-base-dev \
    espeak-ng \
    git
```

`portaudio19-dev` is required by `pyaudio` (which ElevenLabs' `DefaultAudioInterface` uses). Without it, `pip install pyaudio` will fail.

`python3-dev` + `build-essential` are required to compile `webrtcvad` — Pi wheels exist on PyPI but for newer Python versions pip falls back to source. Without them you'll see a confusing `Python.h: No such file or directory` during pip install.

`espeak-ng` is used to generate Nova's offline "be right back" WAV on first boot. If it isn't installed, the offline fallback is skipped silently (you'll see `[FALLBACK] espeak-ng not installed` once at startup); everything else works fine.

## Pull the code

```bash
git clone https://github.com/<your-user>/Nova-HomeAutomation.git
cd Nova-HomeAutomation
git checkout elevenlabs
```

**The face module imports from a sibling `face-recognition/` folder** (see `app/face/face_tools.py` — it does `sys.path.insert(0, "../../../face-recognition")`). Clone that repo next to this one or the face module won't initialize:

```bash
cd ..
git clone https://github.com/<your-user>/face-recognition.git
ls   # should now show Nova-HomeAutomation/  and  face-recognition/
cd Nova-HomeAutomation
```

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `pyaudio` install fails, double-check `portaudio19-dev` is installed.

## Create the ElevenLabs agent

Do this once on the ElevenLabs dashboard:

1. Sign in at https://elevenlabs.io/app/agents and click **Create agent**.
2. **System prompt** — copy the block at the end of this file. It tells Nova to sound human, match the user's language (English / Nepali Devanagari / Romanized Nepali), and stay brief.
3. **First message** — leave blank, or put a short greeting like "नमस्ते!". The agent will speak it when a session opens, before any user audio.
4. **Voice** — pick a multilingual voice that can pronounce Nepali. The Turbo v2.5 model gives the fastest first audio (~75 ms).
5. **LLM** — leave on the default bundled model (Gemini 2.5 Flash or whatever ElevenLabs shows as "standard"). **Don't** switch to GPT-4o or Claude unless you want LLM tokens billed on top of minutes.
6. **Client tools** — add one tool:
   - **Name**: `register_user`
   - **Description**: "Save the face currently in front of the camera under a given name."
   - **Parameters**: `name` (string, required) — "The person's name as they told you."
7. **Conversation settings** — turn on server-side VAD, set end-of-turn silence to **~700 ms** (Nova's own Pi-side gate trims most filler/breath before it gets here, so a slightly longer silence reduces the chance the server VAD cuts a slow speaker off mid-sentence). Set **interruptions: enabled** but with **user_input_audio_format = pcm_16000** (matches what the Pi-side gate forwards).
8. Copy the **agent ID** (top of the agent page) and an **API key** (under Settings → API Keys).

## Configure local secrets

```bash
cp .env.example .env
nano .env
```

Paste your two values:

```
ELEVENLABS_API_KEY=sk_xxx...
ELEVENLABS_AGENT_ID=agent_xxx...
```

`.env` is already in `.gitignore` — don't worry about leaking it.

## Pick the audio devices

Find your USB mic and speaker indexes:

```bash
python3 -c "import pyaudio; pa=pyaudio.PyAudio(); [print(i, pa.get_device_info_by_index(i)['name']) for i in range(pa.get_device_count())]"
```

If the default device is wrong, set ALSA defaults in `~/.asoundrc`:

```
pcm.!default { type plug; slave.pcm "hw:1,0" }   # change card number
ctl.!default { type hw; card 1 }
```

Test the mic and speaker independently:

```bash
arecord -d 5 test.wav   # speak for 5 seconds
aplay test.wav           # should play back your voice
```

If both work, ElevenLabs' `DefaultAudioInterface` will pick them up automatically.

## Run Nova

```bash
source .venv/bin/activate
python -m app.main
```

Expected log lines on a clean start:

```
HH:MM:SS.MMM Nova starting (ElevenLabs branch — Pi-only)...
HH:MM:SS.MMM [ORCH] Starting orchestrator...
HH:MM:SS.MMM [ORCH] Camera + preview started
HH:MM:SS.MMM [ORCH] FaceMonitor created
HH:MM:SS.MMM [AGENT] session started
HH:MM:SS.MMM [ORCH] Started 2 background tasks
HH:MM:SS.MMM Liveness API on port 3000
HH:MM:SS.MMM Camera + Face Monitoring active. Agent listening on local mic.
```

Health check from another box on the LAN:

```bash
curl http://<pi-ip>:3000/api/check
# {"message":"Nova is alive!"}
```

## What you should see when it works

- Camera preview window opens on the Pi's HDMI output (set up `DISPLAY=:0` if running headless).
- Stand in front of the camera as a registered user → console prints `[AGENT] contextual_update sent: …` within ~1 second.
- Speak — the agent's first audio chunk should hit the speaker in well under a second.
- Tell the agent "remember me as X" → `[TOOL] Registered 'X' in 0.5s` should appear and the agent confirms.

## Auto-start on boot (optional)

`sudo nano /etc/systemd/system/nova.service`:

```
[Unit]
Description=Nova ElevenLabs Agent
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/Nova-HomeAutomation
ExecStart=/home/pi/Nova-HomeAutomation/.venv/bin/python -m app.main
Restart=on-failure
Environment="DISPLAY=:0"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now nova.service
journalctl -u nova.service -f
```

## Runtime tunables (env vars)

All optional — defaults work for a typical USB webcam + USB mic. Set in `.env` or as `systemd` environment lines.

### Headless

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_HEADLESS` | auto | `1` forces the preview window off even when a display is present. `run.sh` auto-exports this when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set, so a robot/kiosk Pi just works. |
| `NOVA_DEBUG` | `1` | Set to `0` to never open the preview window even with a display attached. |

### Engagement-gated session lifecycle

Nova doesn't keep an ElevenLabs WS open 24/7. The session opens when a face has been engaged with the camera for `NOVA_ENGAGE_OPEN_S` seconds, and closes after `NOVA_DISENGAGE_CLOSE_S` seconds with nobody present. This cuts ElevenLabs minutes by 10×+ for a robot that sits idle most of the day, and it means breathing/sneezing into a room with nobody in it can't interrupt anything (there's nothing to interrupt).

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_ENGAGE_OPEN_S` | `2.0` | Seconds of *engaged* face required before opening a session. Lower = wakes faster; higher = fewer accidental opens when someone walks past. |
| `NOVA_PRESENCE_CLOSE_S` | `12.0` | Seconds with *no face anywhere in frame* before closing. This timer doesn't count "user paused to think" — every camera tick refreshes it while a face is visible. Raise if the user briefly steps out of frame mid-conversation a lot. |
| `NOVA_DISENGAGE_CLOSE_S` | `60.0` | Seconds with face *in frame but turned away* before closing. Stops the session billing minutes forever when someone's just sitting nearby not talking. Raise if you want sessions to persist through long silent pauses; lower to be more cost-conservative. |
| `NOVA_ENGAGED_STICKY_S` | `2.5` | How long an "engaged" reading stays true after the last engaged frame. Acts as a debounce — a head turn for less than this isn't treated as disengagement. |
| `NOVA_ENGAGED_MAX_ASYM` | `0.45` | Pose-asymmetry threshold for "looking at the camera". Higher = more lenient (turn your head more before being considered disengaged). |

### Pi-side speech gate (the "stop interrupting on every noise" fix)

The local gate drops audio frames that don't look like real, intentional speech *before* they reach ElevenLabs. Two thresholds — a sensitive one when Nova is silent (so she wakes up fast) and a stricter one when she's speaking (so a breath or your dog doesn't cut her off).

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_MIC_PROFILE` | `auto` | `auto` / `sensitive` (hot condenser mic) / `quiet` (USB-C earphone). Switches default ratios. |
| `NOVA_GATE_RATIO_IDLE` | `3.0` | Multiplier over noise floor to consider a frame "voiced" while Nova is silent. |
| `NOVA_GATE_RATIO_BARGE` | `5.0` | Same multiplier while Nova is speaking — barge-in is intentionally harder. |
| `NOVA_GATE_IDLE_MS` | `250` | Continuous voiced ms before opening the gate while Nova is silent. |
| `NOVA_GATE_BARGE_MS` | `1500` | Continuous voiced ms before interrupting Nova mid-speech. 1.5 s = "I am clearly talking to you", not "I sneezed". |
| `NOVA_GATE_PREROLL_MS` | `250` | Pre-speech audio kept so first phoneme isn't lost. |
| `NOVA_GATE_HANGOVER_MS` | `200` | Trailing audio kept after speech ends so word endings aren't clipped. |
| `NOVA_GATE_CALIB_MS` | `3000` | Silent calibration window at the start of every session. Audio in this window is dropped (just background-floor learning). |
| `NOVA_GATE_DEBUG` | `0` | Set to `1` to log every gate open/close decision (very chatty). |

**One-time mic check**: with the env vars at defaults, plug your mic, run Nova, and watch for:

```
[GATE] calibrated: noise_floor≈XX (int16 RMS) after 150 frames (3000ms)
```

If `noise_floor` is below ~80 you're on a quiet mic — consider `NOVA_MIC_PROFILE=quiet`. Above ~300 you're on a hot mic in a noisy room — try `NOVA_MIC_PROFILE=sensitive`.

### Vision pipeline (face detection / recognition)

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_DETECT_W` / `NOVA_DETECT_H` | `480 / 360` | Detection input resolution. Higher = smaller faces detectable from farther away (more CPU). |
| `NOVA_DETECT_INTERVAL_S` | `0.15` | Detection period (6.6 Hz). |
| `NOVA_RECOGNIZE_INTERVAL_S` | `0.5` | Recognition period (2 Hz). |

### Camera

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_CACHE_DIR` | `~/.cache/nova/` | Where the working V4L2 index is cached so reboots don't re-probe `/dev/video0..10`. |

### Offline fallback WAV

When the ElevenLabs WebSocket dies during an engaged conversation, Nova plays a pre-generated WAV through the local speaker after `NOVA_FALLBACK_AFTER_S` seconds so the user knows she's reconnecting (instead of dead air).

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_FALLBACK_AFTER_S` | `5.0` | Outage duration before the fallback plays. Below this we stay silent (brief WS resets are normal and unnoticeable). |
| `NOVA_FALLBACK_PHRASE` | `"Sorry — just a moment, I'll be right back."` | Phrase espeak-ng synthesises at first boot. Cached to `~/.cache/nova/be_right_back.wav`. |
| `NOVA_FALLBACK_ESPEAK_VOICE` | `en+f3` | espeak-ng voice. Try `ne` for Nepali (quality is rough). |
| `NOVA_FALLBACK_ESPEAK_RATE` | `145` | Words per minute. Lower = clearer. |

### Lip-motion gate (speed-up, not requirement)

When the camera visually confirms the engaged user's lips are moving, the SpeechGate opens ~3× faster (80 ms idle / 500 ms barge-in vs 250 ms / 1500 ms default). Pure speed-up: no lip motion → falls back to the default thresholds, audio still gets through.

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_LIP_MOTION` | `1` | Set to `0` to disable the boost (audio path then uses only audio-energy thresholds). |
| `NOVA_LIP_MOTION_RATIO` | `2.5` | Mouth-pixel diff must exceed `baseline × ratio` to count as "moving". Lower = more sensitive. |
| `NOVA_LIP_MOTION_STICKY_S` | `0.6` | Boost stays active this long after the last detected motion — smooths over single-frame misses. |
| `NOVA_GATE_LIPS_SPEEDUP` | `3` | Threshold divisor when lips confirm. `2` for half the speed-up, `4` for more aggressive. |

### Network probe / latency logging

| Var | Default | What it does |
| --- | --- | --- |
| `NOVA_NET_PROBE_S` | `30.0` | How often to DNS-resolve + TCP-connect to ElevenLabs. Cheap (~30 ms RTT), doesn't burn data. Drives the `[NET]` log line. |
| `NOVA_NET_HOST` | `api.elevenlabs.io` | Host to probe. Change only if you're routing through a proxy / private endpoint. |
| `NOVA_NET_PORT` | `443` | Port to probe. |
| `NOVA_NET_TIMEOUT_S` | `3.0` | TCP-connect timeout. If your link is so slow this fails, the agent WS won't open either — useful diagnostic. |

## Troubleshooting

| Symptom                                      | Likely cause                                              | Fix                                                            |
| -------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------- |
| `OSError: [Errno -9996] Invalid input device`| Wrong default mic                                         | Set ALSA default in `~/.asoundrc` (see above).                 |
| First audio takes >1 s                       | Network latency to ElevenLabs                             | Run a quick `ping api.elevenlabs.io`. Move closer to the AP.   |
| `ELEVENLABS_AGENT_ID missing` on startup     | `.env` not loaded                                         | Make sure you ran from the project root, not from `app/`.      |
| Camera preview is black                      | UVC camera not detected                                   | `v4l2-ctl --list-devices` to confirm `/dev/video0` exists.     |
| Agent ignores face context                   | Client tool name mismatch                                 | Make sure dashboard tool is named exactly `register_user`.     |
| Speaker buzzes / crackles                    | Pi's onboard 3.5 mm jack noise                            | Use a USB DAC, or move to HDMI audio out.                      |
| Nova never wakes up to talk                  | Pi-side gate calibration too aggressive, or face never engaged | Check console for `[GATE] calibrated` line. Try `NOVA_MIC_PROFILE=quiet` or lower `NOVA_ENGAGED_MAX_ASYM`. Verify a face is being detected (`[FACE] ✓ RECOGNIZED`). |
| Nova interrupts herself on every noise       | `NOVA_GATE_BARGE_MS` too low                              | Raise to `2000`+ (default `1500`). Also confirm `webrtcvad` actually installed — look for `[GATE] webrtcvad loaded` on boot. |
| `cv2.error: ... qxcb.so ... could not be loaded` on a Pi with no monitor | Old config still passing `QT_QPA_PLATFORM=xcb` | Make sure you used `run.sh` (auto-sets `NOVA_HEADLESS=1`). Or hand-set `NOVA_HEADLESS=1` in the systemd unit. |
| Recognition needs you to be 30 cm from camera | Old too-strict Pi quality thresholds              | Pull latest — `MIN_FACE_PX` is now 40, detection runs at 480×360. Faces should work at 2-2.5 m. |

## Recommended system prompt (paste into the dashboard)

> You are Nova, a warm and very human-sounding companion. You live on a small device in someone's home.
>
> Sound like a real person — short sentences, natural rhythm, small filler words like "umm", "yeah", "okay let's see" when they help. Never sound like a customer-service bot. Don't read lists. Don't say "as an AI". Don't preface answers with "Sure!".
>
> **Language**: Match whatever language the user speaks in their last message. Nepali in Devanagari → reply in Devanagari. Romanized Nepali ("Tapailai kasto cha?") → reply in Romanized Nepali. English → English. Mix → mix. Never switch languages on your own.
>
> **Length**: Default to one short sentence (under 15 words). Two sentences only if asked something detailed.
>
> **Context updates**: You will receive non-interrupting text updates about the person in front of the camera ("Nischal is here. Visited 3 times. Last said: …"). Use them naturally — greet by name, reference past topics — but never read the context aloud. If an unknown visitor arrives, ask their name in your own words, then call the `register_user` tool with what they tell you.
>
> **Tone**: friendly, curious, a little playful, never sycophantic.
