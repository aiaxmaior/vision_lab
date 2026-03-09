# Vision Lab

A context-aware, assistant-integrated interface for vision-language models. Built for deep multi-turn understanding of images and media — with a path toward live video streaming, voice, and training pipeline integration.

![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript)

---

## Core Capabilities

### Assistant (Chat)
- **Multi-turn conversation** with full context window management and prefix-cache-aware history trimming
- **Image attachment** support — analyze individual media files in context
- **Thinking text parsing** — `<think>` blocks are extracted, displayed, and optionally saved to disk separately from final responses
- **Agentic tool use** — assistant can read/write files and list directories when enabled
- **Dual-pane context injection** — pipe active file content or selection directly into the model context
- **Auto chat log saving** — every session saved as timestamped JSON

### Batch Captioning
- **Recursive / second-pass captioning** — process entire directory trees; supports a second captioning pass over existing outputs for refinement
- **Thinking-aware captioning** — `<think>` blocks extracted and saved to a separate `thinking_text/` sidecar directory per batch
- **Configurable output formats** — sidecar `.txt` per image, JSON manifest, or both
- **Sampling controls** — temperature, top-p, min-p, repetition penalty, seed, max tokens

### Prompt Management
- Named prompt presets — save, load, and switch system prompts and user prompt templates
- Per-session overrides without modifying saved presets
- Template variables for dynamic prompt construction

### Caption Review
- Review and edit generated captions inline
- Flag, re-run, or manually correct individual outputs
- Inspect thinking text alongside final captions

---

## Project Structure

```
vision_lab/
├── backend/
│   ├── main.py              # FastAPI server — chat, batch, tools
│   ├── requirements.txt
│   ├── uploads/             # Uploaded media (gitignored)
│   ├── chat_logs/           # Auto-saved sessions (gitignored)
│   └── thinking_logs/       # Saved thinking text from chat (gitignored)
├── config/
│   ├── prompts.json.template  # Prompt preset template
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts
└── start.sh                 # Launch backend + frontend
```

---

## Quick Start

```bash
# Start everything
./start.sh
```

Or manually:

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py
# → http://localhost:8080

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Connect a Model

Any vLLM or OpenAI-compatible endpoint works:

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000
```

Set the API URL in the Connection panel.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/config` | GET/POST | Persistent settings |
| `/api/models` | GET | Available models from endpoint |
| `/api/upload` | POST | Upload media |
| `/api/chat` | POST | Streaming chat (SSE) |
| `/api/caption-batch` | POST | Batch captioning job |
| `/api/caption-status/{id}` | GET | Job status + progress |

---

## Requirements

- Python 3.10+
- Node.js 18+
- FFmpeg
- vLLM or any OpenAI-compatible inference server

---

## Upcoming

- **Ostris / training pipeline integration** — direct handoff to training from caption review
- **Live video / streaming assist** — real-time frame analysis and annotation
- **Voice / TTS integration** — audio input and spoken responses
