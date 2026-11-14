# Rackneh's JaJaJa(rvis)

A local, always-on voice AI assistant with a "conscious" foreground self and a
background task workforce. Jarvis listens for its wake word, talks back with a
cloned Jarvis voice, and runs multi-hour / multi-day projects in the background
while staying fully conversational.

Built on the proven patterns of the Dory LLM / IBA backend (SQL-durable agent
state, GPU-exclusive Dory↔LM Studio switching) and replacing the WLM desktop
voice client with a professional streaming pipeline.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
  microphone ──────────▶│  VOICE PIPELINE (all local)              │
                        │  openWakeWord ("hey jarvis")  ~2 ms/frame│
                        │  Silero VAD (endpointing)                │
                        │  faster-whisper / Moonshine STT (stream) │
                        │  XTTS-v2 Jarvis clone ▸ Kokoro ▸ Piper ▸ │
                        └───────────────┬──────────────────────────┘
                                        │ events (bus)
        ┌───────────────────────────────┼──────────────────────────────┐
        │  CONSCIOUS AGENT (foreground) │  TASK ENGINE (background)    │
        │  streams Dory / LM Studio     │  workers per task, sandboxed │
        │  tools: spawn_task, remember, │  checkpointed steps (SQLite) │
        │  switch_backend, rename…      │  ask_user → relayed to you   │
        │  onboarding interview +       │  multi-hour/day, resumes on  │
        │  background profile parser    │  restart                     │
        └───────────────┬───────────────┴──────────────┬───────────────┘
                        │ WebSocket /ws + REST         │
                ┌───────▼──────────────────────────────▼───────┐
                │  JARVISDECK — WPF neon HUD                   │
                │  voice orb, conscious stream, task cards,    │
                │  question relay, backend switch, mic toggle  │
                └──────────────────────────────────────────────┘
```

### Voice — what changed vs the old WLM client
| Old (WLMRevivalV2Desktop) | Jarvis |
|---|---|
| WAV files + blocking `HttpWebRequest` | in-memory PCM frames, `asyncio` end-to-end |
| re-running Whisper every 650 ms on growing audio | VAD + capped 4 s-tail partials |
| guessing the utterance with a 2nd LLM session, wrong answers audible | no speculation — finalize on Silero endpointing |
| RMS toy VAD | Silero VAD (ONNX, ~2 ms/frame) |
| no wake word | openWakeWord pretrained **"hey jarvis"** (custom names fall back to transcript name-spotting, or push-to-talk) |
| polling endpoints (500 ms loops) | single WebSocket event bus |
| SAPI fallback after first error | engine chain: **jarvis clone → Kokoro → Piper → SAPI** |

### The Jarvis voice
`voice.tts_engine` defaults to `jarvis_clone`. `scripts/build_jarvis_voice.py`
pulls the [`derekurban2001/jarvis-voice-samples`](https://huggingface.co/datasets/derekurban2001/jarvis-voice-samples)
corpus (720 Jarvis-style clips) and builds a 30 s reference WAV. XTTS-v2
(`coqui-tts`, installed by the default launcher) then zero-shot clones that
voice. If cloning is unavailable the chain falls back to Kokoro/Piper/SAPI.

### Model switching (LM Studio ↔ Dory)
Same GPU-exclusive contract as the IBA backend:
- the default provider is **LM Studio** (`providers.default: "lmstudio"` in
  `config.json`); `auto` also probes LM Studio first and falls back to Dory
- switching to Dory unloads LM Studio models and starts the llama.cpp engine
  through the Dory GUI wrapper (`:5000`), and vice versa
- `DynamicProvider` re-resolves per turn, so switches apply at the next step
- exposed as a conscious-agent tool: just say *"switch to Dory"*

### Background tasks
- Task = objective + own workspace dir (`jarvis_home/tasks/task_<id>/`)
- Worker thread steps the task: each step is a fresh prompt assembled from
  durable state (objective, receipts, file listing, user answers) — the model
  replies with one JSON outcome (`act / decompose / ask_user / wait / finish`)
- Everything is checkpointed in SQLite; **tasks survive restarts** and resume
  on boot
- `ask_user` pauses the task, relays the question to you through the conscious
  agent (voice + HUD amber card), and your answer is injected into its next
  step — ask *"how's task X going?"* any time for a progress report
- Parallel workers (default 2) with a shared inference semaphore so the
  foreground conversation always stays snappy

### First boot
The conscious agent runs a natural interview (what to call you, your role,
projects + paths, preferences, permissions). Each answer is parsed into the
profile **in the background while you keep talking**, and an environment scan
indexes your stated project folders. Progress is shown in the HUD. Rename the
assistant any time: *"change your name to Friday"* (HUD top bar follows).

---

## Setup

Requirements: Windows 10/11, Python 3.12, .NET 9 SDK. Dory backend from
`Dory LLM` repo (llama-server + GUI) for the Dory provider; LM Studio running
its local server on `:1234` for the other.

```powershell
# full launch (creates venv, installs deps, downloads voice models, builds the
# Jarvis clone reference, installs coqui-tts/XTTS, starts backend + HUD)
.\scripts\launch_jarvis.ps1

# skip the heavy XTTS/coqui install (Kokoro/Piper/SAPI fallback remains)
.\scripts\launch_jarvis.ps1 -NoVoiceClone

# headless console chat instead of the HUD
.\scripts\launch_jarvis.ps1 -NoGui

.\scripts\stop_jarvis.ps1
```

Config lives in `config.json` (assistant name, wake word, ports, engines,
task parallelism). Persistent state lives in `jarvis_home/` (SQLite, task
workspaces, voice models).

## Using it
- Say **"hey Jarvis"**, then talk. Jarvis stays idle otherwise.
- During speech: *"start a background task: <objective>"* — it confirms once
  and returns to conversation.
- Task questions arrive as spoken relays + amber HUD cards; type or speak the
  answer.
- Any time: *"progress report on <task>"*, *"pause task X"*, *"switch to Dory"*.
- Typed input works identically in the HUD's bottom bar.

## Layout
```
backend/run.py                 entry (server or --cli console)
backend/jarvis/config.py       config.json access
backend/jarvis/events.py       async event bus (one WS channel)
backend/jarvis/providers/      Dory / LM Studio / dynamic + switch_backend
backend/jarvis/memory/         SQLite store + profile
backend/jarvis/conscious/      foreground agent + tools
backend/jarvis/tasks/engine.py background task engine
backend/jarvis/onboarding/     interview + background parser + env scan
backend/jarvis/voice/          wake/VAD/STT/TTS/session state machine
native-gui/                    JarvisDeck WPF HUD (net9.0-windows)
scripts/                       launch/stop/voice-build
```
