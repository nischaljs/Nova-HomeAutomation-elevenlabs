# ElevenLabs Pricing — Nova `elevenlabs` Branch

Numbers pulled from https://elevenlabs.io/pricing/agents on 2026-05-11.

## Plan tiers (Agents / Conversational AI)

| Plan      | Monthly                  | Included agent minutes | Overage  |
| --------- | ------------------------ | ---------------------- | -------- |
| Free      | $0                       | 15 min                 | —        |
| Starter   | $6                       | 75 min                 | $0.08/min |
| Creator   | $22 (first month $11)    | 275 min                | $0.08/min |
| Pro       | $99                      | 1,238 min              | $0.08/min |
| Scale     | $299                     | 3,738 min              | $0.08/min |
| Business  | $990                     | 12,375 min             | $0.08/min |

We are on the **Creator plan ($22/month)**.

## What the per-minute rate covers

A "minute" billed by ElevenLabs Agents bundles **all three stages** of the voice loop:

1. ASR (speech-to-text)
2. LLM reasoning (using the bundled standard model — Gemini 2.5 Flash or equivalent)
3. TTS (text-to-speech) with the voice configured on the agent

Sticking to the bundled standard LLM keeps LLM token cost at **$0**. Upgrading to GPT-4o-mini or Claude in the dashboard charges tokens on top of minutes (deducted from your credit balance).

## Extra charges

- **Text messages** — every `conversation.send_contextual_update(...)` call costs **$0.003**. Nova sends one update on `face_recognized`, `face_unknown`, and `face_lost`. The `FacePresenceTracker` debounces face events upstream, and `ElevenLabsAgent.push_face_context` adds a 2-second throttle on duplicate text — so contextual updates stay rare.
- **Burst pricing** — $0.16/min when you exceed the plan's concurrent-call limit (Creator = 4 concurrent). One Pi can't trigger this.
- **Premium LLMs** — only if you switch the agent's LLM in the dashboard. Stay on the bundled standard model to avoid this line item.

## Projected monthly cost for this project

| Scenario                | Minutes used | Base | Overage           | Contextual updates  | **Monthly total**         |
| ----------------------- | ------------ | ---- | ----------------- | ------------------- | ------------------------- |
| Light home use          | 200          | $22  | $0                | ~$1                 | **~$23**                  |
| Mid (exhibition demo)   | 600          | $22  | $26 (325 × $0.08) | ~$3                 | **~$51**                  |
| Heavy daily use         | 1,200        | $22  | $74               | ~$6                 | **~$102** — switch to Pro |

If sustained usage trends above ~1,000 minutes per month, the **Pro plan ($99 / 1,238 min)** becomes the cheaper option.

## How to keep cost low

- Leave the agent's LLM on the bundled standard model. Free LLM tokens.
- Don't pull the agent into a chatty system prompt — short replies use less TTS time.
- Trust the upstream face debounce. Don't loosen `STABLE_DETECTION_SECONDS` in the face module just to send more contextual updates.
- Keep the 2-second throttle in `ElevenLabsAgent.push_face_context` — it suppresses duplicate text messages when a face is wobbling at the recognition threshold.
- Check usage at https://elevenlabs.io/app/usage weekly.
