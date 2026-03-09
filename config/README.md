# Vision Lab v4.0 — Configuration & Architecture Reference

> **AI Agent Restriction:** Files in this directory (especially `prompts.json`) may contain
> user-authored system prompts with domain-specific content instructions. AI agents MUST
> follow the restrictions defined in [`training_data/AGENTS.md`](../../training_data/AGENTS.md).
> Specifically: **do not read, interpret, or reproduce the content of user-edited `system_prompt`
> values in `config.json` or `prompts.json`.** You may read the structural keys, modify code that
> references them, and create new generic templates — but never inspect or echo back the actual
> prompt content the user has written. When in doubt, ask the user.

## Overview

Vision Lab v4 is a dual-pane interface for working with Vision-Language Models (VLMs) via vLLM. The left pane hosts specialized functions (batch captioning, caption review). The right pane is a persistent chat with the VLM that has contextual awareness of whatever function is active on the left.

The VLM serves **two distinct roles** depending on which path a request takes, and each role has its own system prompt, sampling parameters, and context strategy.

---

## Architecture

```
┌─────────────┐   ┌──────────────────────────────────┐
│   Sidebar    │   │         Main Content Area         │
│  (settings)  │   │                                    │
│              │   │  ┌─────────────┬──────────────┐   │
│  • Connection│   │  │  LEFT PANE  │  RIGHT PANE  │   │
│  • Sampling  │   │  │  (function) │  (chat)      │   │
│  • Media     │   │  │             │              │   │
│  • Inference │   │  │ Batch Cap.  │  Persistent  │   │
│  • Mode      │   │  │ Caption Rev │  VLM Chat    │   │
│              │   │  │             │              │   │
│              │   │  │             │  [Screenshot]│   │
│              │   │  │             │  [Context]   │   │
│              │   │  │             │  [Thinking]  │   │
│              │   │  └─────────────┴──────────────┘   │
└─────────────┘   └──────────────────────────────────┘
                         │                    │
                    Batch API            Chat API
                   (non-streaming)      (streaming SSE)
                         │                    │
                         └────────┬───────────┘
                                  │
                           vLLM Server
                        (Qwen3.5 + BNB)
```

---

## The Two Roles of the VLM

### Role 1: Batch Captioner

**Purpose:** Automated, one-at-a-time captioning of media files. No conversation. One request per file, structured output.

**When it runs:** User points the Batch Caption tab at a directory and clicks Start. Each media file gets a single API call.

**System prompt:** None. The instruction is the entire user message. This keeps the request minimal and focused.

**Instruction source:** `prompts.json` → `batch_captioner.image_instruction` or `batch_captioner.video_instruction`. Can be overridden per-batch via the UI.

**Output format:** Structured sections (`[VISUAL]`, `[SPEECH]`, `[SOUNDS]`, `[TEXT]`). Thinking blocks are stripped before saving.

**Sampling preset:** `captioning` — lower temperature (0.7), moderate presence penalty (0.5), thinking enabled for reasoning but stripped from output.

**Context strategy:** No conversation history. Each file is a fresh, single-turn request. No prefix cache reuse.

**Token budget:**
- Input: ~500 tokens instruction + visual tokens from the media
- Output: up to 4096 tokens (configurable)

---

### Role 2: Chat Assistant ("Meta Eyes")

**Purpose:** Interactive, multi-turn conversation with workspace awareness. Reviews screenshots, critiques captions, guides the user through captioning decisions. Can see what's on the left pane.

**When it runs:** User types in the persistent chat pane on the right. Optionally attaches screenshots of the left pane or uploaded media.

**System prompt:** `prompts.json` → `chat_assistant.system_prompt`. This prompt establishes the assistant's identity as a workstation-integrated reviewer. It knows about LoRA training requirements, caption structure, and visual analysis.

**Why it needs a different prompt than the captioner:** The captioner produces captions. The assistant *evaluates* captions. It needs to understand that its job is to compare a visible caption against visible media and provide corrections, identify hallucinations, flag missed details, and maintain a coherent multi-turn dialogue. It's a reviewer, not a generator.

**Sampling preset:** `thinking_mode` — higher temperature (1.0), high presence penalty (1.5) to reduce repetition across long conversations, thinking enabled.

**Context strategy:**
- **Prefix cache:** Previous messages are included, but **assistant messages have `<think>` blocks stripped**. Only the final response is retained in history. This is critical for token efficiency — thinking blocks can be 500-2000 tokens each, and a 10-turn conversation would otherwise consume 5K-20K tokens on reasoning that the model doesn't need to re-read.
- **Pane context:** When "Context ON" is toggled, a text summary of the left pane state is appended to the system message (not as a separate system message — Qwen3.5 only allows one). This tells the VLM what file is being reviewed, batch progress, current caption text, etc.
- **Screenshots:** When the user clicks the Screenshot button, `html2canvas` captures the left pane as an image. This is sent as a multimodal `image_url` alongside the user's text message. The VLM literally sees the UI — the media being reviewed, the caption text, navigation state, everything visible.

**Token budget:**
- System prompt: up to 1500 tokens
- Pane context: ~100-300 tokens (text summary)
- Screenshot: ~1000-2000 visual tokens (depending on resolution)
- Conversation history (stripped): variable, grows with turns
- Output: up to 16384 tokens to allow extended reasoning

---

## Why Two Separate Prompts

| Aspect | Batch Captioner | Chat Assistant |
|--------|----------------|----------------|
| **Goal** | Generate a caption | Evaluate/improve a caption |
| **Turns** | Single-turn | Multi-turn |
| **Input** | Raw media | Screenshot of media + caption + UI |
| **Output** | Structured sections | Free-form analysis and suggestions |
| **Thinking** | Enabled but stripped from output | Enabled, stripped from history only |
| **System prompt** | None (instruction only) | Persistent identity and role definition |
| **Context** | None | Full conversation + pane state + screenshots |

If you use the chat assistant's system prompt for batch captioning, you get verbose evaluative text instead of clean structured captions. If you use the captioner's instruction for the chat, you lose the conversational identity and workspace awareness.

---

## Context Injection Mechanisms

### 1. Pane Context (Text)
Toggle: "Context" button in chat bar.

Serializes the left pane state as plain text appended to the system message:
- Batch tab: directory path, progress counts, last caption preview
- Caption review tab: current filename, index, whether caption exists, current edit text

Low token cost (~100-300 tokens). Useful for the VLM to know what you're working on without seeing it visually.

### 2. Screenshot (Visual)
Button: "Screenshot" (camera icon) in chat bar.

Captures the entire left pane DOM as a JPEG via `html2canvas` and attaches it as a multimodal image to the next user message. The VLM sees:
- The media being reviewed (rendered in the browser)
- The caption text (rendered in the textarea)
- UI state (navigation, progress bars, etc.)

Higher token cost (~1000-2000 visual tokens) but much richer information. This is the "meta eyes" feature — the VLM can directly compare the visible image against the visible caption text.

### 3. Uploaded Media (Direct)
Standard media upload via sidebar. Sends the actual media file to vLLM via `file://` URL (requires `--allowed-local-media-path /`). This is the raw media, not a screenshot of it.

These three mechanisms can be combined. A typical advanced workflow:
1. Upload media → VLM sees the raw image/video
2. Screenshot → VLM sees the caption that was generated
3. Text prompt → "Is this caption accurate? What did it miss?"
4. Context ON → VLM knows which file index you're on and batch progress

---

## Thinking Mode & Prefix Cache Strategy

Qwen3.5 uses `<think>...</think>` blocks for chain-of-thought reasoning. This is controlled by `enable_thinking` in `chat_template_kwargs`.

**During generation:** Thinking is enabled. The model reasons before responding.

**In conversation history:** Thinking blocks are **stripped server-side** before building the API payload. The assistant's previous messages only contain the final response, not the reasoning chain.

**Why this matters:**
- A typical thinking block is 500-2000 tokens
- A 10-turn conversation accumulates 5-20K tokens of reasoning
- That reasoning is stale — the model generated it for a previous context
- Stripping it saves context budget for actual content (screenshots, media tokens, system prompt)
- With 170K available context after BNB, this keeps conversations viable for 50+ turns

The current turn's thinking is generated normally and visible in the streaming response. It's only stripped when that message becomes history in the *next* turn.

---

## Sampling Presets

Defined in `prompts.json` → `sampling_presets`:

| Preset | Temperature | Top-P | Top-K | Presence | Thinking | Use Case |
|--------|------------|-------|-------|----------|----------|----------|
| `thinking_mode` | 1.0 | 0.95 | 20 | 1.5 | Yes | Chat assistant, general tasks |
| `instruct_mode` | 0.7 | 0.8 | 20 | 1.5 | No | Direct responses, no reasoning |
| `captioning` | 0.7 | 0.9 | 20 | 0.5 | Yes | Batch captioning (structured output) |

The chat bar has a toggle to switch between thinking and instruct mode on the fly.

---

## File Structure

```
vision_lab_v4/
├── config/
│   ├── README.md              ← This file
│   └── prompts.json           ← Prompt templates, sampling presets, token budgets
├── backend/
│   ├── main.py                ← FastAPI server (chat, batch, media, config endpoints)
│   ├── config.json            ← Runtime config (connection, UI state, sampling params)
│   ├── uploads/               ← Uploaded media storage
│   └── chat_logs/             ← Auto-saved conversation logs
├── frontend/
│   ├── src/
│   │   ├── App.tsx            ← Main React component (dual-pane layout)
│   │   ├── main.tsx           ← Entry point
│   │   └── index.css          ← Styles (deep space theme)
│   ├── package.json
│   └── vite.config.ts
├── start.sh
└── README.md                  ← Quick start guide
```

---

## REVEAL Corpus Workflow

The screenshot + chat combination enables building training data for the REVEAL pipeline:

1. Load a directory in Caption Review
2. Navigate to an image
3. Screenshot the pane (VLM sees image + existing caption)
4. Ask the VLM to evaluate the caption's accuracy
5. The VLM reasons through what matches and what doesn't
6. Correct the VLM's reasoning if needed — "look at the jaw tension, that's not a genuine smile"
7. Iterate until the VLM arrives at the correct interpretation
8. Export the full dialogue as JSON (the reasoning corpus button)

The exported JSON captures the multi-turn reasoning trajectory — how the VLM's understanding of the image evolved through guided correction. This is the kind of data needed for training models to reason about visual ambiguity rather than pattern-matching surface features.

---

*Last updated: March 9, 2026*
*Vision Lab v4.0 — ajax*
