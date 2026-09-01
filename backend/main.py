#!/usr/bin/env python3
"""
🎭 VISION LAB v4.0 - FastAPI Backend
Multi-Modal VLM Interface with streaming chat, batch captioning, and Qwen3.5 support.

Title:      Vision Lab Backend API
Author:     ajax
Date:       2026-03-09
Version:    4.0.0
License:    MIT

Description:
    RESTful API backend for Vision Lab, providing endpoints for:
    - Multi-modal chat with VLM models (streaming SSE)
    - Media upload and processing (images/videos)
    - FFmpeg video pre-processing
    - Token estimation for visual content
    - Configuration management

Dependencies:
    - FastAPI, uvicorn, python-multipart
    - OpenCV, Pillow for media processing
    - FFmpeg/FFprobe for video analysis
    - Connects to vLLM server for inference
"""

import cv2
import base64
import requests
import json
import re
import tempfile
import subprocess
import mimetypes
import asyncio
import threading
import os
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image, ImageOps
import io
import shutil
import uuid

app = FastAPI(title="Vision Lab API", version="4.0.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config and storage paths
CONFIG_FILE = Path(__file__).parent / "config.json"
PROMPTS_FILE = Path(__file__).parent.parent / "config" / "prompts.json"
MODES_FILE = Path(__file__).parent.parent / "config" / "modes.yaml"
CUSTOM_MODE_FILE = Path(__file__).parent.parent / "config" / "custom_mode.txt"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CHAT_LOGS_DIR = Path(__file__).parent / "chat_logs"
CHAT_LOGS_DIR.mkdir(exist_ok=True)
THINKING_LOGS_DIR = Path(__file__).parent / "thinking_logs"
THINKING_LOGS_DIR.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model_name": "Qwen35-9B",
    "active_character": "",
    "processing_mode": "Native Video (vLLM)",
    "sampling_mode": "fps",
    "interval": 2.0,
    "target_fps": 1.0,
    "max_frames_limit": 0,
    "resolution_mode": "User Defined",
    "image_width": 640,
    "image_height": 480,
    "system_prompt": "",
    "interaction_mode": "Free-form",
    "custom_mode": False,
    "inject_thinking_tags": False,
    "max_images_in_context": 3,
    "max_tokens": 16384,
    "temperature": 1.0,
    "top_p": 0.95,
    "min_p": 0.0,
    "top_k": 20,
    "repetition_penalty": 1.0,
    # presence/frequency are OpenAI-semantics additive penalties on a -2..2 scale.
    # They are NOT repetition_penalty. Non-zero frequency_penalty crushes the most
    # frequently repeated tokens first — i.e. punctuation and function words — which
    # degenerates long outputs into comma-less run-ons. Default both to 0.0 and use
    # repetition_penalty / min_p to control repetition instead.
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": -1,
    "thought_syntax": "<think>{content}</think>",
    "tool_call_format": "auto",
    "vram_limit": 170000,
    # TTS integration (any OpenAI-compatible TTS server, default: local supertonic+omnivoice on :8800)
    "tts_enabled": False,
    "tts_url": "http://localhost:8800/v1/audio/speech",
    "tts_model": "omnivoice",          # "omnivoice" | "supertonic"
    "tts_voice": "F2",                  # supertonic voice key OR ignored if ref_audio set
    "tts_ref_audio": "",                # path to reference WAV for omnivoice voice cloning
    "tts_instruct": "",                 # voice design string for omnivoice (alt to ref_audio)
    "tts_speed": 1.0,
    "tts_strip_thinking": True,         # strip <think>...</think> before synthesis
    # Two-pass observation (opt-in): when media is attached, the VLM first runs
    # a brief structured observation pass, then its observation is injected as
    # silent context into the system message of the actual response generation.
    # Inspired by REVEAL's "make the signal explicit and inspectable" principle.
    "enable_observation_pass": False,
    # The currently active observation prompt. Swap this string with one of the
    # variants in `observation_prompt_examples` below (or fine-tune your own)
    # to change pass-A behavior. Default = neutral prose (good for RP + general).
    "observation_prompt": (
        "Describe what is visible in the attached media in 4-6 sentences of plain "
        "neutral prose. Be specific about subjects (named colors, positions, postures), "
        "setting (location, lighting, time of day), and activity. Don't interpret, "
        "don't ask questions, don't speculate about meaning or emotion — just describe "
        "what's there as if dictating into a notebook. If something is unclear due to "
        "occlusion, framing, or quality, say so plainly."
    ),
    # --- Live proactive screen awareness (the OBS/SAY loop) ---
    # One merged call per cycle: the model describes the screen (OBS) AND decides
    # whether to speak up unprompted (SAY), given the recent conversation + a short
    # frame clip. Kept short and cheap; cooldown prevents chatter.
    "live_turn_prompt": (
        "You are silently watching the user's screen during an ongoing conversation. "
        "You receive the recent conversation, then a short clip (frames in chronological "
        "order) of what is on screen right now. Do two things:\n"
        "1) OBS: in ONE neutral sentence, state what is currently on screen.\n"
        "2) SAY: decide if something on screen is DIRECTLY relevant to the conversation "
        "— e.g. the user asked you to watch for something and it just happened. If and "
        "only if it clearly matters, write ONE concise sentence (<=25 words) addressed to "
        "the user. Otherwise write exactly: [SILENT]\n"
        "Do not narrate generally. Do not repeat a point you already made. Only speak when "
        "it matters. Output exactly two lines, the first starting with \"OBS:\" and the "
        "second starting with \"SAY:\"."
    ),
    "live_turn_max_tokens": 96,
    "live_turn_temperature": 0.4,
    "live_cooldown_seconds": 15,
    "live_clip_frames": 8,
    "live_clip_span_ms": 3500,
    # Reference variants — copy one into `observation_prompt` above to switch.
    # The leading underscore signals "documentation / not loaded as live config."
    "_observation_prompt_examples": {
        "neutral_prose": (
            "Describe what is visible in the attached media in 4-6 sentences of plain "
            "neutral prose. Be specific about subjects, setting, and activity. Don't "
            "interpret or speculate — just describe what's there. State limitations plainly."
        ),
        "structured_forensic": (
            "Take a careful structured observation pass. Produce ONLY a concise "
            "observation block — do not answer any question yet. Format:\n"
            "[SUBJECTS] who/what is present (entities, count, relationships)\n"
            "[SETTING] where, when, lighting/conditions, framing\n"
            "[ACTIVITY] what is happening\n"
            "[NOTABLE] non-obvious details that might be relevant to follow-up questions\n"
            "[UNCERTAIN] anything you cannot reliably observe (occlusion, framing, quality)\n"
            "Be specific — name colors, positions, postures, text. Avoid generalities."
        ),
        "cinematic_third_person": (
            "Describe the attached media as if writing a single paragraph of stage "
            "direction for a screenplay. Use evocative but concrete language: actual "
            "colors, postures, gaze direction, light quality, spatial relationships. "
            "No dialogue, no interpretation of motive, no questions. 4-7 sentences."
        ),
        "first_person_perception": (
            "You are about to relay what is visible in the attached media to someone "
            "who cannot see it. Speak in the present tense, as if standing there yourself. "
            "Lead with the most immediate impression, then fill in details: subjects, "
            "setting, activity, anything notable. 3-6 sentences. Don't editorialize."
        ),
        "technical_quantitative": (
            "Produce a technical inventory of the attached media. List:\n"
            "- Composition: framing (wide/medium/close), camera angle, depth of field\n"
            "- Subjects: count, position (e.g., left-third foreground), bounding-box "
            "estimate of dominant subject as fraction of frame\n"
            "- Lighting: source direction, hard/soft, color temperature estimate\n"
            "- Notable objects with approximate location\n"
            "- Quality artifacts (motion blur, compression, focus issues)\n"
            "Be quantitative where possible. Skip interpretation."
        ),
        "terse_factual": (
            "List 5-8 short factual observations about the attached media, one per line, "
            "no preamble, no interpretation. Examples of granularity: 'two people seated', "
            "'wooden table center frame', 'overcast daylight from left'."
        ),
        "rp_character_voice_hint": (
            "Describe what is visible in the attached media in plain present-tense prose, "
            "as if you were the perceiver looking at the scene right now. Lead with the "
            "most striking element. Be specific (colors, positions, expressions) without "
            "interpreting emotion or asking questions. 4-6 sentences. The text you produce "
            "will become silent perceptual context for an in-character response — write "
            "it so a character could read it as their own immediate impression."
        ),
    },
    "observation_max_tokens": 1024,
    "observation_temperature": 0.4,
    "observation_include_media_in_pass_b": True,  # if False, pass B sees only the observation text, not the media (cheaper)
    "effort_level": "xhigh",
    # --- Agentic web tools (web_search / fetch_url) ---
    "search_provider": "duckduckgo",   # "duckduckgo" | "searxng" | "tavily" | "brave"
    "search_max_results": 5,
    "searxng_url": "",                 # e.g. http://localhost:8888  (SearXNG instance with JSON API enabled)
    "tavily_api_key": "",
    "brave_api_key": "",
    "fetch_url_max_chars": 8000,        # cap on fetch_url text length to protect context budget

}


def load_prompts() -> dict:
    """Load prompt templates from config/prompts.json."""
    if PROMPTS_FILE.exists():
        try:
            with open(PROMPTS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_custom_instructions() -> str:
    """Load the optional custom-mode system preamble from config/custom_mode.txt.

    Local/personal file, not tracked in the repo. Returns "" when absent, in
    which case custom mode is a no-op.
    """
    if CUSTOM_MODE_FILE.exists():
        try:
            return CUSTOM_MODE_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


UI_MODE_KEYS = {
    "Free-form": "free_form",
    "Analytical": "analytical",
    "Roleplay": "roleplay",
}


def load_modes() -> dict:
    """Load UI-agent prompt profiles from config/modes.yaml.

    Returns {"ui_modes": {free_form|analytical|roleplay: {text_prompt,
    observation_prompt, media_prompt: {image, video}, ...}}} or
    {"ui_modes": {}} on any failure. Prompts live in YAML so multi-line
    content stays readable.
    """
    if MODES_FILE.exists():
        try:
            with open(MODES_FILE, 'r') as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and "ui_modes" in data:
                return data
        except Exception:
            pass
    return {"ui_modes": {}}


def save_modes(data: dict):
    """Persist the full ui_modes object back to config/modes.yaml."""
    MODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODES_FILE, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False)


def resolve_ui_mode(interaction_mode: str, active_character: str = "") -> dict:
    """Resolve the prompt bundle for a given interaction mode.

    Returns {text_prompt, observation_prompt, live_turn_prompt, media_image, media_video}.
    For Roleplay, the selected character's text_prompt (if any) overrides the
    mode-level text_prompt.
    """
    ui_modes = load_modes().get("ui_modes", {})
    key = UI_MODE_KEYS.get(interaction_mode, "free_form")
    mode = ui_modes.get(key, {}) or {}
    media = mode.get("media_prompt", {}) or {}
    bundle = {
        "text_prompt": mode.get("text_prompt", "") or "",
        "observation_prompt": mode.get("observation_prompt", "") or "",
        "live_turn_prompt": mode.get("live_turn_prompt", "") or "",
        "media_image": media.get("image", "") or "",
        "media_video": media.get("video", "") or "",
    }
    if key == "roleplay":
        chars = mode.get("characters", {}) or {}
        char_name = active_character or mode.get("active_character", "") or ""
        char = chars.get(char_name, {}) or {}
        if isinstance(char, dict) and char.get("text_prompt"):
            bundle["text_prompt"] = char["text_prompt"]
    return bundle


def normalize_reasoning_channels(content: str) -> str:
    """Convert gemma4's leaked channel markup into <think>...</think> blocks.

    The abliterated gemma4 model emits chain-of-thought inside the content field
    as malformed harmony-style channel tokens, e.g.
        <|channel>thought\\n...reasoning...<channel|>answer
    vLLM does not route this to the OpenAI `reasoning` field, so without this the
    markup leaks raw into the visible answer and into saved history. Rewrite
    well-formed channel pairs to <think>...</think> (which the UI collapses and
    the history stripper recognizes) and drop any stray/unterminated markers.
    """
    if not isinstance(content, str) or "<|channel>" not in content:
        return content
    import re

    def _repl(m):
        inner = m.group(1).strip()
        return f"<think>{inner}</think>" if inner else ""

    content = re.sub(r"<\|channel>\w*\s*(.*?)<channel\|>", _repl, content, flags=re.DOTALL)
    # Drop any leftover/unterminated markers (e.g. response truncated by max_tokens)
    content = re.sub(r"<\|channel>\w*\s*", "", content)
    content = content.replace("<channel|>", "")
    return content.strip()


def strip_thinking_from_content(content: str) -> str:
    """Remove <think>...</think> blocks from assistant messages for prefix cache efficiency.

    Only the final response is retained in conversation history.
    This saves significant context tokens during multi-turn conversations.
    """
    if not isinstance(content, str):
        return content
    content = normalize_reasoning_channels(content)
    lower = content.lower()
    idx = lower.find("</think>")
    if idx != -1:
        return content[idx + len("</think>"):].strip()
    import re
    return re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()

INTERACTION_MODES = {
    "Free-form": {
        "description": "No system constraints - pure prompt passthrough",
        "inject_system": False,
        "inject_thinking": False
    },
    "Analytical": {
        "description": "Structured analysis with optional thinking tags",
        "inject_system": True,
        "inject_thinking": True
    },
    "Roleplay": {
        "description": "Character/scenario mode - uses system prompt as character definition",
        "inject_system": True,
        "inject_thinking": False
    }
}

# --- Tool Definitions for Agentic Chat ---

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file. Use this to inspect captions, configs, or any text file the user references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the text file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command and return the output. Use this to execute system commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite the contents of a text file. Use this to update captions, save corrections, or create new text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the text file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory. Optionally filter by extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory"
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g. '.txt', '.json')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a list of results (title, url, snippet). Use to find current information, documentation, or facts not in your training data. Follow up with fetch_url to read a result in full.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page and return its text content with HTML stripped. Use after web_search to read a result, or to read any known http(s) URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http(s) URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_media",
            "description": "Look at an image or video file and get a detailed text description of its visible content. Use this to inspect screenshots, photos, or video clips the user references. Accepts the same sandboxed paths as read_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the image or video file"
                    }
                },
                "required": ["path"]
            }
        }
    }
]

# Allowed base directories for tool file operations (safety constraint)
TOOL_ALLOWED_PATHS = [
    Path("/media/ajax/AI"),
#    Path("/home/ajax"),
]


def is_path_allowed(file_path: str) -> bool:
    """Check if a file path is within allowed directories."""
    resolved = Path(file_path).resolve()
    return any(resolved.is_relative_to(base) for base in TOOL_ALLOWED_PATHS)


def _web_search(query: str, max_results, config: dict) -> dict:
    """Pluggable web search. Provider selected by config['search_provider'].

    Supports: duckduckgo (default, needs the `ddgs` pip package), searxng
    (config['searxng_url']), tavily (config['tavily_api_key']), brave
    (config['brave_api_key']). Returns {query, provider, results:[{title,url,snippet}]}.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "Empty search query"}
    provider = (config.get("search_provider") or "duckduckgo").lower()
    try:
        n = int(max_results) if max_results else int(config.get("search_max_results", 5) or 5)
    except (TypeError, ValueError):
        n = 5
    n = max(1, min(n, 25))

    try:
        if provider == "duckduckgo":
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    return {"error": "DuckDuckGo search needs the 'ddgs' package. Install with: pip install ddgs"}
            hits = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=n):
                    hits.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", "") or r.get("url", ""),
                        "snippet": r.get("body", "") or r.get("snippet", ""),
                    })
            return {"query": query, "provider": provider, "results": hits}

        if provider == "searxng":
            base = (config.get("searxng_url") or "").rstrip("/")
            if not base:
                return {"error": "Set 'searxng_url' in config.json to use the SearXNG provider"}
            resp = requests.get(f"{base}/search", params={"q": query, "format": "json"}, timeout=20)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            hits = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                    for r in results[:n]]
            return {"query": query, "provider": provider, "results": hits}

        if provider == "tavily":
            key = config.get("tavily_api_key") or ""
            if not key:
                return {"error": "Set 'tavily_api_key' in config.json to use the Tavily provider"}
            resp = requests.post("https://api.tavily.com/search",
                                 json={"api_key": key, "query": query, "max_results": n}, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            hits = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                    for r in results[:n]]
            return {"query": query, "provider": provider, "results": hits}

        if provider == "brave":
            key = config.get("brave_api_key") or ""
            if not key:
                return {"error": "Set 'brave_api_key' in config.json to use the Brave provider"}
            resp = requests.get("https://api.search.brave.com/res/v1/web/search",
                                params={"q": query, "count": n},
                                headers={"X-Subscription-Token": key, "Accept": "application/json"}, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("web", {}).get("results", [])
            hits = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
                    for r in results[:n]]
            return {"query": query, "provider": provider, "results": hits}

        return {"error": f"Unknown search_provider: {provider}"}
    except Exception as e:
        return {"error": f"web_search failed ({provider}): {e}"}


def _fetch_url(url: str, config: dict) -> dict:
    """Fetch a URL and return text with HTML stripped, capped to config['fetch_url_max_chars']."""
    url = (url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return {"error": "url must start with http:// or https://"}
    try:
        max_chars = int(config.get("fetch_url_max_chars", 8000) or 8000)
    except (TypeError, ValueError):
        max_chars = 8000
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (VisionLab agent)"})
        resp.raise_for_status()
        text = resp.text
        if "html" in resp.headers.get("Content-Type", "").lower():
            import re
            text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "content": text[:max_chars], "length": len(text), "truncated": len(text) > max_chars}
    except Exception as e:
        return {"error": f"fetch_url failed: {e}"}


VIEW_MEDIA_INSTRUCTION = (
    "Describe the attached media in clear, factual detail so it can be reasoned "
    "about: subjects, setting, any visible text, actions, colors, and notable "
    "details. For video, summarize what happens across the frames. Describe only "
    "what is visible; do not speculate. Output only the description."
)


def describe_media_file(media_path: str, config: dict) -> dict:
    """Run a VLM pass over an image/video file and return a text description.

    Backs the `view_media` agent tool: tool results travel back to the model as
    text, so the agent 'reads' media by getting a faithful description rather
    than raw pixels.
    """
    media_type = get_media_type(media_path)
    if media_type not in ("image", "video"):
        return {"error": f"Not an image or video file: {media_path}"}
    try:
        media_content = prepare_media_content(
            media_path,
            "",                    # processing_mode: frame-sample video (not Native Video)
            "interval",            # sampling_mode
            2.0,                   # interval (seconds between sampled frames)
            1.0,                   # target_fps (unused in interval mode)
            8,                     # max_frames_limit
            640, 480,              # frame size for video; ignored for native images
            "Native Resolution",   # images full-res; video frames at 640x480
        )
        if not media_content:
            return {"error": "Could not prepare media content"}
        payload = {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": VIEW_MEDIA_INSTRUCTION},
                {"role": "user", "content": [{"type": "text", "text": "Describe this media."}] + media_content},
            ],
            "max_tokens": int(config.get("observation_max_tokens", 1024)),
            "temperature": float(config.get("observation_temperature", 0.4)),
            "stream": False,
        }
        resp = requests.post(config["api_url"], json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"].get("content", "") or ""
        return {
            "path": media_path,
            "media_type": media_type,
            "description": strip_thinking_from_content(raw).strip(),
        }
    except Exception as e:
        return {"error": str(e)[:500]}


VIDEO_DIGEST_INSTRUCTION = (
    "You are compressing a video into a durable note that must survive after the "
    "video itself leaves the context window. Reply with at most five short lines:\n"
    "1. One sentence on what the video shows overall.\n"
    "2-4. The key moments, each as 'mm:ss - what happens'.\n"
    "5. Any visible text, labels, or identifiers worth recalling verbatim.\n"
    "Describe only what is visible. Do not speculate. Output only those lines."
)

DIGEST_DIR = UPLOAD_DIR / "_digests"


def _video_digest(media_path: str, media_id: str, config: dict) -> str:
    """One-off text digest of a video, cached on disk by media_id.

    Computed lazily the first time a video falls out of the replay window, so
    videos that are never referenced again cost nothing. `_upload_paths()` walks
    UPLOAD_DIR for files only, so this subdirectory stays invisible to it.
    """
    DIGEST_DIR.mkdir(exist_ok=True)
    cached = DIGEST_DIR / f"{media_id}.txt"
    if cached.exists():
        try:
            return cached.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    try:
        frames = prepare_media_content(
            media_path, "", "interval", 2.0, 1.0, 8, 640, 480, "Native Resolution",
        )
        if not frames:
            return ""
        resp = requests.post(config["api_url"], json={
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": VIDEO_DIGEST_INSTRUCTION},
                {"role": "user", "content": [{"type": "text", "text": "Digest this video."}] + frames},
            ],
            "max_tokens": 400,
            "temperature": 0.3,
            "stream": False,
        }, timeout=180)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"].get("content", "") or ""
        text = strip_thinking_from_content(raw).strip()
        if text:
            try:
                cached.write_text(text, encoding="utf-8")
            except OSError:
                pass
        return text
    except Exception as e:
        print(f"[digest] failed for {media_id}: {str(e)[:200]}")
        return ""


def _extract_keyframes(video_path: str, n: int, size: tuple = (640, 480)) -> List[str]:
    """N evenly-spaced frames spanning the whole video, as base64 JPEGs.

    extract_frames_manual reads sequentially and stops at max_frames, so asking
    it for 3 frames returns the first three sampled — the opening moment, not a
    summary. This seeks instead, so the frames actually span the runtime.
    """
    if n <= 0:
        return []
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        # Sample at the midpoints of n equal slices, so we never land on a black
        # first frame or a truncated last one.
        targets = [int(total * (2 * k + 1) / (2 * n)) for k in range(n)]
        out = []
        for idx in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
            ok, frame = cap.read()
            if not ok:
                continue
            resized = cv2.resize(frame, (int(size[0]), int(size[1])))
            _, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            out.append(base64.b64encode(buf).decode("utf-8"))
        return out
    finally:
        cap.release()


def _video_fallback_content(media_path: str, media_id: str, config: dict,
                            n_frames: int) -> List[dict]:
    """Degraded stand-in for a video that is out of the replay budget.

    A few evenly-spaced keyframes give the model something real to look at, and
    the cached digest carries the temporal detail that stills cannot. Either half
    may come back empty; whatever survives is better than the bare note.
    """
    out: List[dict] = []
    try:
        for b64 in _extract_keyframes(media_path, n_frames):
            out.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    except Exception as e:
        print(f"[fallback] keyframes failed for {media_id}: {str(e)[:200]}")

    digest = _video_digest(media_path, media_id, config)
    label = (
        "[The full video is no longer in the visual context window. "
        f"{'Above are ' + str(len(out)) + ' sampled keyframes. ' if out else ''}"
        "Summary recorded while it was visible:]\n" + digest
    ) if digest else (
        "[The full video is no longer in the visual context window"
        + (f"; above are {len(out)} sampled keyframes.]" if out else
           ". Ask the user to re-attach it if you need to look at it again.]")
    )
    out.append({"type": "text", "text": label})
    return out


def execute_tool(name: str, arguments: dict, config: Optional[dict] = None) -> dict:
    """Execute a tool call and return the result."""
    config = config or {}
    if name == "read_file":
        path = arguments["path"]
        if not is_path_allowed(path):
            return {"error": f"Access denied: {path} is outside allowed directories"}
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        if not p.is_file():
            return {"error": f"Not a file: {path}"}
        try:
            content = p.read_text(encoding="utf-8")
            return {"content": content, "size": len(content), "path": str(p)}
        except Exception as e:
            return {"error": f"Failed to read {path}: {e}"}

    elif name == "write_file":
        path = arguments["path"]
        content = arguments["content"]
        if not is_path_allowed(path):
            return {"error": f"Access denied: {path} is outside allowed directories"}
        p = Path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {"status": "written", "path": str(p), "size": len(content)}
        except Exception as e:
            return {"error": f"Failed to write {path}: {e}"}

    elif name == "list_directory":
        path = arguments["path"]
        ext = arguments.get("extension")
        if not is_path_allowed(path):
            return {"error": f"Access denied: {path} is outside allowed directories"}
        p = Path(path)
        if not p.is_dir():
            return {"error": f"Not a directory: {path}"}
        try:
            entries = sorted(p.iterdir())
            if ext:
                entries = [e for e in entries if e.suffix.lower() == ext.lower()]
            files = [{"name": e.name, "is_dir": e.is_dir(), "size": e.stat().st_size if e.is_file() else 0}
                     for e in entries[:500]]  # cap at 500 entries
            return {"path": str(p), "count": len(files), "entries": files}
        except Exception as e:
            return {"error": f"Failed to list {path}: {e}"}

    elif name == "view_media":
        path = arguments["path"]
        if not is_path_allowed(path):
            return {"error": f"Access denied: {path} is outside allowed directories"}
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        if not p.is_file():
            return {"error": f"Not a file: {path}"}
        return describe_media_file(str(p), config)

    elif name == "web_search":
        return _web_search(arguments.get("query", ""), arguments.get("max_results"), config)

    elif name == "fetch_url":
        return _fetch_url(arguments.get("url", ""), config)

    elif name == "run_shell_command":
        command = arguments.get("command", "")
        if not command:
            return {"error": "No command provided"}
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            return {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except Exception as e:
            return {"error": f"Failed to run command: {e}"}

    return {"error": f"Unknown tool: {name}"}


CUSTOM_INSTRUCTIONS = load_custom_instructions()


# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be string or list for multimodal
    # Upload attached to this turn. Lets the backend re-attach the image on later
    # turns so the model can look at it again instead of only ever seeing it on
    # the turn it arrived in.
    media_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    media_path: Optional[str] = None
    include_media: Optional[bool] = False
    # Inference settings
    max_tokens: int = 40960
    temperature: float = 1.0
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = -1
    # How many turns' images stay in the visual context window (most recent win).
    max_images_in_context: int = 3
    # Videos get their own budget: in Native Video mode a past video replays as a
    # single video_url block, but vLLM expands it server-side, so its token cost
    # is unlike an image's and counting it against max_images_in_context would
    # badly under-count. 0 disables video replay entirely.
    max_videos_in_context: int = 1
    # Videos past that budget degrade instead of vanishing: a few keyframes plus a
    # cached one-off text digest. 0 frames falls back to the digest alone.
    video_fallback_frames: int = 3
    # Qwen3.5 thinking mode
    enable_thinking: bool = False
    # Two-pass observation override (per-request; falls back to config.json default)
    enable_observation_pass: Optional[bool] = None
    # Mode settings
    interaction_mode: str = "Free-form"
    active_character: str = ""
    system_prompt: str = ""
    inject_thinking: bool = False
    custom_mode: bool = False
    thought_syntax: str = "<think>{content}</think>"
    # Media settings
    processing_mode: str = "Native Video (vLLM)"
    sampling_mode: str = "fps"
    interval: float = 2.0
    target_fps: float = 1.0
    max_frames_limit: int = 0
    resolution_mode: str = "User Defined"
    image_width: int = 640
    image_height: int = 480
    video_fps: float = 2.0
    force_fps: bool = True
    # Context injection from left pane
    pane_context: Optional[str] = None
    # Latest live screen observation (from /api/observe-frame). Injected as
    # situated context so the user can talk about what's on screen right now.
    live_observation: Optional[str] = None
    # Agentic tool use
    tools_enabled: bool = False
    # How to read the model's tool calls: "server" trusts vLLM's --tool-call-parser
    # only; the others recover the call from message content when that parser and
    # the model disagree. See TOOL_CALL_FORMATS.
    tool_call_format: str = "auto"


class BatchCaptionRequest(BaseModel):
    directory: str
    instruction: str = ""
    caption_target: str = ""
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    enable_thinking: bool = False
    video_fps: float = 2.0
    force_fps: bool = True
    strip_thinking: bool = True
    skip_existing: bool = True
    output_format: str = "json"
    # Optional assistant-prefill: seed the model's reply with the first tokens of
    # the caption so it continues in the right register instead of opening with a
    # hedge/euphemism. Falls back to prompts.json batch_captioner.prefill if None.
    caption_prefill: Optional[str] = None


class BatchStopRequest(BaseModel):
    pass


class ConfigUpdate(BaseModel):
    config: Dict[str, Any]


class ProcessVideoRequest(BaseModel):
    media_id: str
    start_time: float
    end_time: float
    width: int
    height: int


# Utility Functions
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config_file(config: dict):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def is_image_file(file_path: str) -> bool:
    if not file_path:
        return False
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('image/')


def is_video_file(file_path: str) -> bool:
    if not file_path:
        return False
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('video/')


def get_media_type(file_path: str) -> Optional[str]:
    if is_image_file(file_path):
        return 'image'
    elif is_video_file(file_path):
        return 'video'
    return None


def get_image_info(image_path: str) -> dict:
    if not image_path:
        return {}
    try:
        img = Image.open(image_path)
        width, height = img.size
        return {"res": f"{width}x{height}", "width": width, "height": height, "type": "image"}
    except:
        return {}


def get_video_info(video_path: str) -> dict:
    if not video_path:
        return {}
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    video.release()
    
    info = {
        "fps": fps, "frames": total_frames, "res": f"{width}x{height}",
        "width": width, "height": height, "dur": duration, "type": "video",
        "video_codec": "Unknown", "audio_codec": "None", "audio_channels": 0,
        "audio_sample_rate": 0, "bitrate": 0
    }
    
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["video_codec"] = stream.get("codec_name", "Unknown").upper()
                    if "bit_rate" in stream:
                        info["video_bitrate"] = int(stream["bit_rate"]) // 1000
                elif stream.get("codec_type") == "audio":
                    info["audio_codec"] = stream.get("codec_name", "Unknown").upper()
                    info["audio_channels"] = stream.get("channels", 0)
                    info["audio_sample_rate"] = int(stream.get("sample_rate", 0))
            fmt = probe_data.get("format", {})
            if "bit_rate" in fmt:
                info["bitrate"] = int(fmt["bit_rate"]) // 1000
    except:
        pass
    
    return info


def get_media_info(file_path: str) -> dict:
    media_type = get_media_type(file_path)
    if media_type == 'image':
        info = get_image_info(file_path)
        info['media_type'] = 'image'
        return info
    elif media_type == 'video':
        info = get_video_info(file_path)
        info['media_type'] = 'video'
        return info
    return {}


def process_image_to_base64(image_path: str, width: int = None, height: int = None) -> str:
    img = Image.open(image_path)
    if width and height:
        img = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def extract_frames_manual(video_path: str, sampling_mode: str, interval: float, 
                          target_fps: float, max_frames: int, size: tuple) -> List[str]:
    """Extract frames and return as base64 strings"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_b64 = []
    step = max(1, int(fps / target_fps)) if sampling_mode == "fps" else max(1, int(fps * interval))
    current = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if current % step == 0:
            resized = cv2.resize(frame, (int(size[0]), int(size[1])))
            _, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frames_b64.append(base64.b64encode(buf).decode('utf-8'))
        current += 1
        if max_frames and len(frames_b64) >= int(max_frames):
            break
    
    video.release()
    return frames_b64


def build_system_message(interaction_mode: str, system_prompt: str, thought_syntax: str,
                         inject_thinking: bool, custom_mode: bool = False) -> Optional[str]:
    mode_config = INTERACTION_MODES.get(interaction_mode, INTERACTION_MODES["Free-form"])

    if not mode_config["inject_system"]:
        # Free-form: passthrough, but inject the mode's text_prompt (and the
        # custom preamble) when present.
        parts = []
        if custom_mode and CUSTOM_INSTRUCTIONS:
            parts.append(CUSTOM_INSTRUCTIONS.strip())
        if system_prompt:
            parts.append(system_prompt)
        return "\n\n".join(parts) if parts else None

    thinking_instruction = ""
    if inject_thinking and mode_config["inject_thinking"] and thought_syntax and "{content}" in thought_syntax:
        open_tag = thought_syntax.split("{content}")[0]
        close_tag = thought_syntax.split("{content}")[1]
        thinking_instruction = f"\n\nUse {open_tag} and {close_tag} tags for your internal reasoning before responding."
    
    custom_prefix = CUSTOM_INSTRUCTIONS if custom_mode else ""

    if interaction_mode == "Roleplay":
        if system_prompt:
            return custom_prefix + system_prompt + thinking_instruction
        elif custom_mode and CUSTOM_INSTRUCTIONS:
            return CUSTOM_INSTRUCTIONS.strip() + thinking_instruction
        return None

    if interaction_mode == "Analytical":
        base = system_prompt if system_prompt else "Provide detailed, structured analysis of the media content."
        return custom_prefix + base + thinking_instruction
    
    return None


def prepare_media_content(media_path: str, processing_mode: str, sampling_mode: str,
                          interval: float, target_fps: float, max_frames_limit: int,
                          image_width: int, image_height: int, resolution_mode: str) -> List[dict]:
    """Prepare media content for API request"""
    media_type = get_media_type(media_path)
    content_list = []
    
    use_native = resolution_mode == "Native Resolution"
    effective_width = None if use_native else image_width
    effective_height = None if use_native else image_height
    
    if media_type == 'image':
        b64 = process_image_to_base64(media_path, effective_width, effective_height)
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    
    elif media_type == 'video':
        if "Native Video" in processing_mode:
            content_list.append({
                "type": "video_url",
                "video_url": {"url": f"file://{media_path}"}
            })
        else:
            frame_width = image_width if not use_native else 640
            frame_height = image_height if not use_native else 480
            frames_b64 = extract_frames_manual(
                media_path, sampling_mode, interval, target_fps,
                max_frames_limit, (frame_width, frame_height)
            )
            for b64 in frames_b64:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
    
    return content_list


def calculate_token_estimate(media_info: dict, target_fps: float, image_width: int, 
                             image_height: int, resolution_mode: str, 
                             context_limit: int, max_model_len: int = None) -> dict:
    """Calculate token estimates for the media"""
    limit = max_model_len if max_model_len else context_limit
    limit_source = "model" if max_model_len else "config"
    
    media_type = media_info.get('media_type')
    if not media_type:
        return {"status": "no_media", "message": "Upload media to estimate..."}
    
    # Determine effective dimensions
    if resolution_mode == "Native Resolution":
        w = media_info.get('width', 640)
        h = media_info.get('height', 480)
    else:
        w = image_width or 640
        h = image_height or 480
    
    tokens_per_frame = (w * h) / 784
    
    if media_type == 'image':
        total_visual_tokens = tokens_per_frame
    else:
        dur = media_info.get('dur', 0)
        total_visual_tokens = (dur * target_fps / 2) * tokens_per_frame
    
    remaining = limit - total_visual_tokens
    status = "good" if remaining > 20000 else "warning" if remaining > 0 else "danger"
    
    return {
        "status": status,
        "visual_tokens": int(total_visual_tokens),
        "remaining": int(remaining),
        "context_limit": limit,
        "limit_source": limit_source,
        "media_type": media_type
    }


# ---------- TTS proxy ----------
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    model: Optional[str] = None
    ref_audio: Optional[str] = None
    instruct: Optional[str] = None
    speed: Optional[float] = None
    response_format: str = "wav"


@app.post("/api/tts")
async def tts_proxy(req: TTSRequest):
    """Synthesize speech via the configured TTS server. Proxies to keep
    secrets/config out of the frontend and to apply server-side defaults
    (voice, ref_audio, etc.) from config.json.
    """
    config = load_config()
    if not config.get("tts_enabled", False):
        raise HTTPException(status_code=400, detail="TTS is disabled in config (tts_enabled=false)")

    text = req.text or ""
    if config.get("tts_strip_thinking", True):
        text = strip_thinking_from_content(text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text after stripping thinking blocks")

    # Build payload — request fields override config defaults
    payload = {
        "model": req.model or config.get("tts_model", "omnivoice"),
        "input": text,
        "voice": req.voice or config.get("tts_voice", "F2"),
        "speed": req.speed if req.speed is not None else float(config.get("tts_speed", 1.0)),
        "response_format": req.response_format,
    }
    ref = req.ref_audio if req.ref_audio is not None else config.get("tts_ref_audio", "")
    if ref:
        payload["ref_audio"] = ref
    instr = req.instruct if req.instruct is not None else config.get("tts_instruct", "")
    if instr:
        payload["instruct"] = instr

    url = config.get("tts_url", "http://localhost:8800/v1/audio/speech")
    try:
        # Long timeout — synthesis on long messages can take 30-60s
        upstream = requests.post(url, json=payload, timeout=300, stream=True)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"TTS server unreachable at {url}: {e}")

    if upstream.status_code != 200:
        raise HTTPException(
            status_code=upstream.status_code,
            detail=f"TTS upstream error: {upstream.text[:500]}",
        )

    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(
        req.response_format, "audio/wav"
    )

    def iter_audio():
        for chunk in upstream.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return StreamingResponse(iter_audio(), media_type=mime)


@app.get("/api/tts/health")
async def tts_health():
    """Probe the configured TTS server for readiness."""
    config = load_config()
    if not config.get("tts_enabled", False):
        return {"enabled": False, "status": "disabled"}
    url = config.get("tts_url", "http://localhost:8800/v1/audio/speech")
    base = url.rsplit("/v1/", 1)[0]
    try:
        r = requests.get(f"{base}/health", timeout=3)
        return {"enabled": True, "status": "ready" if r.ok else "error",
                 "upstream": r.json() if r.ok else r.text[:200]}
    except Exception as e:
        return {"enabled": True, "status": "unreachable", "error": str(e)}


# API Endpoints
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "4.0.0"}


@app.get("/api/prompts")
async def get_prompts():
    """Return prompt templates for the frontend to display/edit."""
    return load_prompts()


@app.get("/api/caption-targets")
async def get_caption_targets():
    """Return available caption targets for the dropdown."""
    prompts = load_prompts()
    targets = prompts.get("caption_targets", {})
    result = []
    for key, target in targets.items():
        if key.startswith("_"):
            continue
        result.append({
            "id": key,
            "name": target.get("name", key),
            "description": target.get("description", ""),
            "style": target.get("style", "nlp"),
            "token_limit": target.get("token_limit"),
            "media_types": target.get("media_types", ["image", "video"]),
        })
    return {"targets": result}


@app.post("/api/prompts")
async def save_prompts(update: Dict[str, Any]):
    """Save updated prompt templates to the active profile."""
    try:
        PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROMPTS_FILE, 'w') as f:
            json.dump(update, f, indent=2, ensure_ascii=False)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Prompt Profile Management ---

PROMPTS_DIR = PROMPTS_FILE.parent


@app.get("/api/prompt-profiles")
async def list_prompt_profiles():
    """List all saved prompt profiles in the config directory."""
    profiles = []
    if PROMPTS_DIR.exists():
        for f in sorted(PROMPTS_DIR.glob("prompts*.json")):
            name = f.stem
            display = "Default" if name == "prompts" else name.replace("prompts_", "").replace("_", " ")
            is_default = name == "prompts"
            try:
                with open(f, 'r') as fh:
                    data = json.load(fh)
                desc = data.get("_description", "")
            except Exception:
                desc = ""
            profiles.append({
                "filename": f.name,
                "name": display,
                "is_default": is_default,
                "description": desc,
            })
    return {"profiles": profiles}


def _vllm_base() -> str:
    """Origin of the vLLM server, derived from the configured chat endpoint."""
    return load_config()["api_url"].split("/v1/")[0]


@app.get("/api/sleep-status")
async def sleep_status():
    """Whether the model is currently offloaded, plus free VRAM for display.

    Requires the server to be started with --enable-sleep-mode and
    VLLM_SERVER_DEV_MODE=1; without both, /is_sleeping 404s and the UI should
    hide the toggle rather than offer a control that cannot work.
    """
    try:
        r = requests.get(f"{_vllm_base()}/is_sleeping", timeout=5)
        if r.status_code == 404:
            return {"available": False, "reason": "sleep mode not enabled on this server"}
        r.raise_for_status()
        sleeping = bool(r.json().get("is_sleeping"))
    except Exception as e:
        return {"available": False, "reason": str(e)[:200]}

    free_gb = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            free_gb = round(sum(int(x) for x in out.stdout.split()) / 1024, 1)
    except Exception:
        pass
    return {"available": True, "is_sleeping": sleeping, "free_vram_gb": free_gb}


@app.post("/api/sleep")
async def sleep_model(level: int = Query(1, ge=1, le=2)):
    """Offload the model so other workloads can use the GPU.

    Level 1 moves weights to CPU RAM — wake is fast, but the host must hold the
    weights. Level 2 discards them and reloads from disk on wake: frees RAM too,
    but waking costs a full load. Level 1 is the right default for flipping back
    and forth with e.g. Diffusion work.
    """
    try:
        r = requests.post(f"{_vllm_base()}/sleep", params={"level": level}, timeout=120)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"sleep failed: {str(e)[:300]}")
    return await sleep_status()


@app.post("/api/wake")
async def wake_model():
    """Bring the model back onto the GPU. Level-2 sleeps reload from disk."""
    try:
        r = requests.post(f"{_vllm_base()}/wake_up", timeout=600)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"wake failed: {str(e)[:300]}")
    return await sleep_status()


@app.get("/api/prompt-profiles/{filename}")
async def load_prompt_profile(filename: str):
    """Load a specific prompt profile."""
    filepath = PROMPTS_DIR / filename
    if not filepath.exists() or not filepath.name.startswith("prompts"):
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaveProfileRequest(BaseModel):
    filename: str
    data: Dict[str, Any]


@app.post("/api/prompt-profiles/save")
async def save_prompt_profile(req: SaveProfileRequest):
    """Save a prompt profile to a named file."""
    safe_name = req.filename.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Filename required")
    if not safe_name.startswith("prompts"):
        safe_name = f"prompts_{safe_name}"
    if not safe_name.endswith(".json"):
        safe_name += ".json"
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe_name)

    filepath = PROMPTS_DIR / safe_name
    try:
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(req.data, f, indent=2, ensure_ascii=False)
        return {"status": "saved", "filename": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/prompt-profiles/{filename}")
async def delete_prompt_profile(filename: str):
    """Delete a prompt profile (cannot delete default)."""
    if filename == "prompts.json":
        raise HTTPException(status_code=400, detail="Cannot delete the default profile")
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    filepath.unlink()
    return {"status": "deleted"}


@app.post("/api/prompt-profiles/activate/{filename}")
async def activate_prompt_profile(filename: str):
    """Set a profile as the active prompts.json (copies it over the default)."""
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Profile not found")
    if filename == "prompts.json":
        return {"status": "already_active"}
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        with open(PROMPTS_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {"status": "activated", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    return load_config()


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    try:
        current = load_config()
        current.update(update.config)
        save_config_file(current)
        return {"status": "success", "message": "Configuration saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ui-modes")
async def get_ui_modes():
    """Return the full UI-agent prompt set (values included) for the editor."""
    return load_modes()


class UiModesUpdate(BaseModel):
    ui_modes: Dict[str, Any]


@app.post("/api/ui-modes")
async def save_ui_modes(update: UiModesUpdate):
    """Persist the full ui_modes object back to config/modes.yaml."""
    try:
        save_modes({"ui_modes": update.ui_modes})
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models(api_url: str = Query(...)):
    """Fetch available models from vLLM server"""
    try:
        base_url = api_url.split("/v1/")[0]
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json().get('data', [])
            models = [m['id'] for m in data]
            max_model_len = data[0].get('max_model_len') if data else None
            return {
                "models": models,
                "max_model_len": max_model_len,
                "status": "connected"
            }
    except Exception as e:
        return {
            "models": [],
            "max_model_len": None,
            "status": "offline",
            "error": str(e)
        }


@app.post("/api/upload")
async def upload_media(file: UploadFile = File(...)):
    """Upload media file and return its ID and info"""
    try:
        # Generate unique filename
        file_ext = Path(file.filename).suffix
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get media info
        info = get_media_info(str(file_path))
        
        return {
            "id": file_id,
            "filename": file.filename,
            "path": str(file_path),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/media/{media_id}")
async def get_media_endpoint(media_id: str):
    """Get media info by ID"""
    # Find the file
    for f in UPLOAD_DIR.iterdir():
        if f.stem == media_id:
            info = get_media_info(str(f))
            return {"path": str(f), "info": info}
    raise HTTPException(status_code=404, detail="Media not found")


def _upload_paths() -> Dict[str, Path]:
    """media_id -> uploaded file, built once per request.

    UPLOAD_DIR accumulates hundreds of files, so resolving ids one at a time by
    rescanning the directory is quadratic over a long conversation.
    """
    try:
        return {f.stem: f for f in UPLOAD_DIR.iterdir() if f.is_file()}
    except OSError:
        return {}


@app.get("/api/media/{media_id}/file")
async def get_media_file(media_id: str):
    """Serve the raw upload so the chat can render it inline.

    /api/media/{id} returns JSON metadata and the thumbnail route returns base64
    inside JSON, so neither can back an <img>/<video> src.

    FileResponse rather than StreamingResponse because it honours Range requests.
    Without that a <video> cannot seek and the browser pulls the whole file just to
    read its metadata — several hundred MB for a long upload.
    """
    for f in UPLOAD_DIR.iterdir():
        if f.stem == media_id:
            mime = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            return FileResponse(f, media_type=mime)
    raise HTTPException(status_code=404, detail="Media not found")


@app.get("/api/media/{media_id}/thumbnail")
async def get_thumbnail(media_id: str, width: int = 320, height: int = 240):
    """Get thumbnail for media"""
    for f in UPLOAD_DIR.iterdir():
        if f.stem == media_id:
            media_type = get_media_type(str(f))
            
            if media_type == 'image':
                b64 = process_image_to_base64(str(f), width, height)
                return {"thumbnail": f"data:image/jpeg;base64,{b64}"}
            
            elif media_type == 'video':
                # Extract first frame
                video = cv2.VideoCapture(str(f))
                success, frame = video.read()
                video.release()
                
                if success:
                    resized = cv2.resize(frame, (width, height))
                    _, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    b64 = base64.b64encode(buf).decode('utf-8')
                    return {"thumbnail": f"data:image/jpeg;base64,{b64}"}
            
            raise HTTPException(status_code=400, detail="Cannot generate thumbnail")
    
    raise HTTPException(status_code=404, detail="Media not found")


@app.post("/api/process-video")
async def process_video(req: ProcessVideoRequest):
    """Process video with FFmpeg"""
    # Find the source file
    source_path = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == req.media_id:
            source_path = str(f)
            break
    
    if not source_path:
        raise HTTPException(status_code=404, detail="Media not found")
    
    if get_media_type(source_path) != 'video':
        raise HTTPException(status_code=400, detail="Not a video file")
    
    # Generate output path
    output_id = str(uuid.uuid4())
    output_path = UPLOAD_DIR / f"{output_id}.mp4"
    
    cmd = [
        "ffmpeg", "-y", "-ss", str(req.start_time), "-to", str(req.end_time),
        "-i", source_path, "-vf", f"scale={req.width}:{req.height}",
        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
        "-c:a", "copy", str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        info = get_media_info(str(output_path))
        return {
            "id": output_id,
            "path": str(output_path),
            "info": info
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr.decode()}")


@app.post("/api/token-estimate")
async def token_estimate(
    media_id: str = Query(None),
    target_fps: float = Query(1.0),
    image_width: int = Query(640),
    image_height: int = Query(480),
    resolution_mode: str = Query("User Defined"),
    context_limit: int = Query(164000),
    max_model_len: int = Query(None)
):
    """Calculate token estimate for media"""
    if not media_id:
        return {"status": "no_media", "message": "Upload media to estimate..."}
    
    # Find the file
    for f in UPLOAD_DIR.iterdir():
        if f.stem == media_id:
            info = get_media_info(str(f))
            return calculate_token_estimate(
                info, target_fps, image_width, image_height,
                resolution_mode, context_limit, max_model_len
            )
    
    return {"status": "no_media", "message": "Media not found"}


def _save_thinking_from_response(full_text: str):
    """Extract thinking from a completed response and save to thinking_logs/."""
    _, thinking = _strip_thinking_tags(full_text)
    if thinking:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        filepath = THINKING_LOGS_DIR / f"chat_{ts}.txt"
        filepath.write_text(thinking, encoding="utf-8")


class ObserveFrameRequest(BaseModel):
    # One captured frame as a data URL (data:image/jpeg;base64,...) or raw base64.
    frame: str
    interaction_mode: str = "Free-form"
    active_character: str = ""
    # Optional explicit prompt override; otherwise the active mode's
    # observation_prompt is used (falling back to DEFAULT_CONFIG).
    observation_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class LiveMsg(BaseModel):
    role: str
    content: str


class LiveTurnRequest(BaseModel):
    # Chronological burst of frames (base64 or data: URIs) — a short "clip".
    frames: List[str]
    interaction_mode: str = "Free-form"
    active_character: str = ""
    # Recent conversation, already bounded client-side; re-clamped here.
    recent_messages: List[LiveMsg] = []
    # Optional prompt override; else the active mode's live_turn_prompt; else DEFAULT_CONFIG.
    live_turn_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # Phase 2 seam: base64 wav or data:audio/wav;base64,... (system audio burst).
    audio: Optional[str] = None


@app.post("/api/observe-frame")
async def observe_frame_endpoint(req: ObserveFrameRequest):
    """Run a single observation pass over one live-captured screen frame.

    Backs live screen sharing: the frontend holds a getDisplayMedia stream open
    and posts frames here back-to-back (paced by model latency, not a timer).
    Returns only the observation text — no chat, no second pass — so the loop
    stays tight. The latest observation is fed back into /api/chat as
    `live_observation`. Pixels travel inline as base64; vLLM never touches disk,
    so no --allowed-local-media-path is required.
    """
    config = load_config()
    mode_bundle = resolve_ui_mode(req.interaction_mode, req.active_character)
    obs_instruction = (
        req.observation_prompt
        or mode_bundle["observation_prompt"]
        or DEFAULT_CONFIG.get("observation_prompt", "")
    )

    frame = req.frame.strip()
    if not frame.startswith("data:"):
        frame = f"data:image/jpeg;base64,{frame}"

    obs_messages = [
        {"role": "system", "content": obs_instruction},
        {"role": "user", "content": [
            {"type": "text", "text": "Observe and report."},
            {"type": "image_url", "image_url": {"url": frame}},
        ]},
    ]
    payload = {
        "model": config["model_name"],
        "messages": obs_messages,
        "max_tokens": int(req.max_tokens or config.get("observation_max_tokens", 1024)),
        "temperature": float(
            req.temperature if req.temperature is not None
            else config.get("observation_temperature", 0.4)
        ),
        "stream": False,
    }
    try:
        resp = requests.post(config["api_url"], json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"].get("content", "") or ""
        return JSONResponse({"observation": strip_thinking_from_content(raw).strip()})
    except Exception as e:
        return JSONResponse({"error": str(e)[:500]}, status_code=502)


# How much recent conversation the live loop may transmit (defensive cap;
# the client already slices before sending).
LIVE_CONTEXT_TURNS = 6
LIVE_MSG_CHAR_CAP = 500


def _parse_live_output(raw: str) -> dict:
    """Parse the two-line OBS:/SAY: contract. Fails safe to silence.

    Expected:
        OBS: <one sentence describing the screen now>
        SAY: <one sentence to the user>   |   SAY: [SILENT]
    A model that drops the prefixes or rambles is treated as observation-only,
    so a malformed cycle never produces an unwanted interjection.
    """
    text = strip_thinking_from_content(raw).strip()
    obs, say, found = "", "", False
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("OBS:"):
            obs = s[4:].strip()
            found = True
        elif s.upper().startswith("SAY:"):
            say = s[4:].strip()
            found = True
    if not found:
        obs = text  # malformed — treat whole output as observation, never speak
    silent = (not say) or say.upper().startswith("[SILENT]") or say.upper() == "SILENT"
    return {
        "observation": obs,
        "interjection": None if silent else say,
        "silent": silent,
    }


@app.post("/api/live-turn")
async def live_turn_endpoint(req: LiveTurnRequest):
    """One proactive 'watch the screen' cycle (the OBS/SAY loop).

    Given a short frame-burst (a clip) plus a bounded slice of the recent
    conversation, the model emits two lines: OBS (a one-sentence description that
    becomes the rolling silent context) and SAY (a concise interjection, or the
    sentinel [SILENT]). One merged call does both — the gate decision and the
    spoken line share the same expensive multimodal prefill, so splitting them
    would only double latency on a single-GPU server. Frames travel inline as
    base64; no disk access. `audio` (Phase 2) is fused as Gemma 4 audio_url.
    """
    config = load_config()
    mode_bundle = resolve_ui_mode(req.interaction_mode, req.active_character)
    instruction = (
        req.live_turn_prompt
        or mode_bundle.get("live_turn_prompt")
        or DEFAULT_CONFIG.get("live_turn_prompt", "")
    )

    # Bound the conversation the loop sees (client already slices; re-clamp here).
    recent = req.recent_messages[-LIVE_CONTEXT_TURNS:]
    transcript = "\n".join(
        f"{m.role}: {m.content[:LIVE_MSG_CHAR_CAP]}" for m in recent
    ) or "(no conversation yet)"

    user_content = [
        {"type": "text",
         "text": f"Recent conversation:\n{transcript}\n\nCurrent screen clip (frames in order):"},
    ]
    for f in req.frames:
        f = f.strip()
        url = f if f.startswith("data:") else f"data:image/jpeg;base64,{f}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    if req.audio:
        a = req.audio.strip()
        a_url = a if a.startswith("data:") else f"data:audio/wav;base64,{a}"
        user_content.append({"type": "audio_url", "audio_url": {"url": a_url}})

    payload = {
        "model": config["model_name"],
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": int(req.max_tokens or config.get("live_turn_max_tokens", 96)),
        "temperature": float(
            req.temperature if req.temperature is not None
            else config.get("live_turn_temperature", 0.4)
        ),
        "stream": False,
    }
    try:
        resp = requests.post(config["api_url"], json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"].get("content", "") or ""
        return JSONResponse(_parse_live_output(raw))
    except Exception as e:
        return JSONResponse({"error": str(e)[:500]}, status_code=502)


# --- Inline tool-call parsing -------------------------------------------------
# vLLM's --tool-call-parser is a launch flag, so swapping models normally means
# restarting the server with a different parser. When the configured parser does
# not match what the model emits, vLLM returns tool_calls=null and the raw call
# markup leaks into the message content. These parsers recover the call from that
# content so the format can be picked per-model from the UI instead.

TOOL_CALL_FORMATS = ("auto", "server", "qwen3_xml", "hermes")

# Openers that mean "a tool call starts here". Matching either lets the XML form
# be recovered whether or not the model wraps it in <tool_call>.
_TOOL_OPENERS = ("<tool_call>", "<function=")

_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_TOOL_FN_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|$)", re.DOTALL)
_TOOL_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)(?:</parameter>|$)", re.DOTALL)


def _tool_param_schema(tool_name: str) -> dict:
    for t in CHAT_TOOLS:
        fn = t.get("function", {})
        if fn.get("name") == tool_name:
            return fn.get("parameters", {}).get("properties", {}) or {}
    return {}


def _coerce_tool_arg(raw: str, schema: dict):
    """XML parameter bodies are always strings; cast them to the declared type.

    vLLM's own parsers do this from the tool schema. Without it, an integer
    parameter arrives as "10" and strict tools reject it.
    """
    value = raw.strip()
    kind = (schema or {}).get("type")
    if kind in ("integer", "number", "boolean", "array", "object"):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _parse_xml_tool_calls(text: str) -> List[dict]:
    """Qwen3 XML: <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>"""
    calls = []
    for name, body in _TOOL_FN_RE.findall(text):
        props = _tool_param_schema(name)
        args = {
            key: _coerce_tool_arg(val, props.get(key, {}))
            for key, val in _TOOL_PARAM_RE.findall(body)
        }
        calls.append({"name": name, "arguments": args})
    return calls


def _parse_hermes_tool_calls(text: str) -> List[dict]:
    """Hermes: <tool_call>{"name": ..., "arguments": {...}}</tool_call>"""
    calls = []
    for block in _TOOL_BLOCK_RE.findall(text):
        try:
            obj = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return calls


def parse_inline_tool_calls(text: str, fmt: str) -> List[dict]:
    """Recover tool calls the server's parser missed. Returns OpenAI-shaped calls."""
    if not text or fmt == "server":
        return []
    if fmt == "hermes":
        parsed = _parse_hermes_tool_calls(text)
    elif fmt == "qwen3_xml":
        parsed = _parse_xml_tool_calls(text)
    else:  # auto — Hermes JSON is unambiguous, so try it before the XML form
        parsed = _parse_hermes_tool_calls(text) or _parse_xml_tool_calls(text)

    return [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
        }
        for i, c in enumerate(parsed)
    ]


class _InlineToolCallFilter:
    """Withholds inline tool-call markup from the user-visible stream.

    When the model writes its call as ordinary content, streaming it verbatim
    shows the user raw XML/JSON. Text is emitted normally until an opener appears,
    then buffered for the parsers. A short tail is always held back so an opener
    split across two SSE chunks is still caught.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.pending = ""
        self.capturing = False
        self.captured: List[str] = []
        self._hold = max(len(o) for o in _TOOL_OPENERS) - 1

    def feed(self, text: str) -> str:
        """Take a content delta; return the portion safe to show the user."""
        if not self.enabled:
            return text
        if not text:
            return ""
        if self.capturing:
            self.captured.append(text)
            return ""

        self.pending += text
        hits = [i for i in (self.pending.find(o) for o in _TOOL_OPENERS) if i != -1]
        if hits:
            idx = min(hits)
            self.capturing = True
            emit = self.pending[:idx]
            self.captured.append(self.pending[idx:])
            self.pending = ""
            return emit

        if len(self.pending) <= self._hold:
            return ""
        emit, self.pending = self.pending[:-self._hold], self.pending[-self._hold:]
        return emit

    def flush(self) -> str:
        """Emit whatever was held back once the round ends without a tool call."""
        if self.capturing:
            return ""
        out, self.pending = self.pending, ""
        return out

    def markup(self) -> str:
        return "".join(self.captured)


async def _sse_payloads(url: str, payload: dict, http_request: Optional[Request] = None):
    """POST a streaming completion and yield each SSE data payload as a string.

    requests' iter_lines() is blocking, so it is pumped from a daemon thread into
    an asyncio queue. Closing the response drops the TCP connection, which is how
    vLLM learns to abort the request when the browser disconnects.
    """
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None, lambda: requests.post(url, json=payload, stream=True, timeout=600)
    )
    resp.raise_for_status()

    queue: asyncio.Queue = asyncio.Queue()

    def _pump():
        try:
            for line in resp.iter_lines():
                loop.call_soon_threadsafe(queue.put_nowait, line)
        except Exception:
            pass
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_pump, daemon=True).start()

    try:
        while True:
            if http_request is not None and await http_request.is_disconnected():
                break
            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if line is None:
                break
            if not line:
                continue
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if decoded.startswith("data: "):
                decoded = decoded[6:]
            if decoded == "[DONE]":
                break
            yield decoded
    finally:
        resp.close()


async def _stream_tool_round(url: str, payload: dict, http_request: Optional[Request] = None,
                             tool_call_format: str = "auto"):
    """Run one streaming round of the tool-use loop.

    Yields ("delta", text) for text that should reach the user immediately, then
    exactly one ("done", {content, reasoning, tool_calls, finish_reason}) with the
    assembled assistant message. Streaming every round — rather than buffering to
    find out whether it contained tool calls — is what lets the final answer stream.
    Tool-call argument fragments still have to be accumulated before they can be
    parsed and executed.
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: Dict[int, dict] = {}
    finish_reason = ""
    think_open = False
    inline = _InlineToolCallFilter(tool_call_format != "server")

    async for decoded in _sse_payloads(url, payload, http_request):
        try:
            chunk = json.loads(decoded)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}

        # vLLM's reasoning parsers route thinking into a separate field; re-wrap it
        # inline so the frontend regex and the thinking_logs save path keep working.
        reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
        if reasoning:
            if not think_open:
                think_open = True
                yield ("delta", "<think>")
            reasoning_parts.append(reasoning)
            yield ("delta", reasoning)

        content = delta.get("content") or ""
        if content:
            if think_open:
                think_open = False
                yield ("delta", "</think>")
            visible = inline.feed(content)
            if visible:
                content_parts.append(visible)
                yield ("delta", visible)

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(idx, {
                "id": "", "type": "function",
                "function": {"name": "", "arguments": ""},
            })
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]

    if think_open:
        yield ("delta", "</think>")

    tail = inline.flush()  # no tool call materialised — release the held-back text
    if tail:
        content_parts.append(tail)
        yield ("delta", tail)

    assembled = []
    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        if not tc["id"]:
            tc["id"] = f"call_{idx}"  # some parsers omit ids when streaming
        assembled.append(tc)

    # Only fall back to inline parsing when the server's parser found nothing.
    inline_markup = inline.markup()
    if not assembled and inline_markup:
        assembled = parse_inline_tool_calls(inline_markup, tool_call_format)
        if not assembled:
            # Looked like a tool call but didn't parse — show it rather than eat it.
            content_parts.append(inline_markup)
            yield ("delta", inline_markup)

    yield ("done", {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": assembled,
        "finish_reason": finish_reason,
    })


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """Streaming chat endpoint"""
    config = load_config()

    async def generate():
        try:
            # Build messages for API
            api_messages = []
            
            # Resolve the system prompt from the UI agent's mode (config/modes.yaml).
            # A non-empty request.system_prompt (session override) wins; otherwise
            # the active mode's text_prompt is used (Roleplay: the selected character).
            mode_bundle = resolve_ui_mode(request.interaction_mode, request.active_character)
            text_prompt = request.system_prompt or mode_bundle["text_prompt"]

            if not request.messages or request.messages[0].role != "system":
                system_msg = build_system_message(
                    request.interaction_mode,
                    text_prompt,
                    request.thought_syntax,
                    request.inject_thinking,
                    request.custom_mode
                )
                if system_msg:
                    api_messages.append({"role": "system", "content": system_msg})
            
            # Re-attach uploads from earlier turns. Previously an upload only reached
            # the model on the turn it was sent — history was rebuilt as plain text —
            # so any follow-up question about the image had nothing to look at and the
            # model would go hunting for a picture that wasn't in its context.
            # max_images_in_context bounds the replay, most recent kept; the current
            # turn's own media counts against that budget.
            uploads = _upload_paths()
            current_turn_images = 1 if (request.include_media and request.media_path) else 0
            history_budget = max(0, request.max_images_in_context - current_turn_images)

            # Images and video are budgeted separately. In Native Video mode a past
            # video replays as one video_url block that vLLM expands server-side, so
            # its cost is nothing like an image's and sharing a budget would
            # under-count it badly. In frame-sample mode each frame is its own
            # base64 block, so replay stays off and video degrades instead.
            native_video = "Native Video" in (request.processing_mode or "")
            img_idx, vid_idx = [], []
            for i, m in enumerate(request.messages):
                if not (m.media_id and m.media_id in uploads):
                    continue
                (vid_idx if is_video_file(str(uploads[m.media_id])) else img_idx).append(i)

            keep_media = set(img_idx[-history_budget:]) if history_budget else set()
            video_budget = request.max_videos_in_context if native_video else 0
            keep_video = set(vid_idx[-video_budget:]) if video_budget else set()
            keep_media |= keep_video
            # Videos that missed the cut still leave a trace rather than vanishing.
            degrade_video = set(vid_idx) - keep_video

            # Process messages, stripping thinking blocks from assistant history
            for i, msg in enumerate(request.messages):
                content = msg.content
                if msg.role == "assistant" and isinstance(content, str):
                    content = strip_thinking_from_content(content)

                if msg.media_id:
                    if i in keep_media:
                        replay = prepare_media_content(
                            str(uploads[msg.media_id]),
                            request.processing_mode,
                            request.sampling_mode,
                            request.interval,
                            request.target_fps,
                            request.max_frames_limit,
                            request.image_width,
                            request.image_height,
                            request.resolution_mode,
                        )
                        if isinstance(content, str):
                            content = ([{"type": "text", "text": content}] if content else []) + replay
                        elif isinstance(content, list):
                            content = content + replay
                    elif i in degrade_video:
                        # Keyframes plus a cached digest, so a video that has aged out
                        # can still be referred back to instead of becoming a dead end.
                        replay = _video_fallback_content(
                            str(uploads[msg.media_id]), msg.media_id, config,
                            request.video_fallback_frames,
                        )
                        if isinstance(content, str):
                            content = ([{"type": "text", "text": content}] if content else []) + replay
                        elif isinstance(content, list):
                            content = content + replay
                    else:
                        # Say so, rather than leaving the model to search for an image
                        # that has silently dropped out of the window.
                        note = ("[Media was attached to this turn but is no longer in the "
                                "visual context window. Ask the user to re-attach it if you "
                                "need to look at it again.]")
                        if isinstance(content, str):
                            content = f"{content}\n\n{note}" if content else note
                        elif isinstance(content, list):
                            content = content + [{"type": "text", "text": note}]

                api_messages.append({"role": msg.role, "content": content})
            
            # Add media to the last user message if requested
            if request.include_media and request.media_path:
                # Find the last user message
                for i in range(len(api_messages) - 1, -1, -1):
                    if api_messages[i]["role"] == "user":
                        # Prepare media content
                        media_content = prepare_media_content(
                            request.media_path,
                            request.processing_mode,
                            request.sampling_mode,
                            request.interval,
                            request.target_fps,
                            request.max_frames_limit,
                            request.image_width,
                            request.image_height,
                            request.resolution_mode
                        )
                        
                        # Convert string content to list format
                        if isinstance(api_messages[i]["content"], str):
                            api_messages[i]["content"] = [
                                {"type": "text", "text": api_messages[i]["content"]}
                            ] + media_content
                        else:
                            api_messages[i]["content"].extend(media_content)
                        break
            
            # Determine whether the two-pass observation is active this turn.
            obs_enabled = (request.enable_observation_pass
                            if request.enable_observation_pass is not None
                            else config.get("enable_observation_pass", False))

            # Media scaffold (Observe OFF): prepend the mode's per-media-type
            # instruction to the user's turn when media is attached.
            if request.include_media and request.media_path and not obs_enabled:
                scaffold = (mode_bundle["media_video"]
                            if is_video_file(request.media_path)
                            else mode_bundle["media_image"])
                if scaffold:
                    for i in range(len(api_messages) - 1, -1, -1):
                        if api_messages[i]["role"] == "user":
                            c = api_messages[i]["content"]
                            if isinstance(c, str):
                                api_messages[i]["content"] = f"{scaffold}\n\n{c}" if c else scaffold
                            elif isinstance(c, list):
                                merged = False
                                for part in c:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        part["text"] = f"{scaffold}\n\n{part.get('text', '')}".strip()
                                        merged = True
                                        break
                                if not merged:
                                    c.insert(0, {"type": "text", "text": scaffold})
                            break

            # ---- Two-pass observation (when enabled + media attached) ----
            # Pass A: short focused observation of the media. Output is injected
            # as silent context into pass B's system message. Optionally surfaced
            # to the UI as a collapsible block.
            observation_text: Optional[str] = None
            if (obs_enabled and request.include_media and request.media_path):
                try:
                    # Prompts never come from backend/config.json. Use the active
                    # mode's observation_prompt (modes.yaml); fall back to the
                    # hardcoded DEFAULT_CONFIG value so a stale config.json can't
                    # inject a prompt here.
                    obs_instruction = mode_bundle["observation_prompt"] or DEFAULT_CONFIG.get("observation_prompt", "")
                    obs_messages = [
                        {"role": "system", "content": obs_instruction},
                    ]
                    # Re-prepare the same media for the observation pass
                    obs_media = prepare_media_content(
                        request.media_path,
                        request.processing_mode,
                        request.sampling_mode,
                        request.interval,
                        request.target_fps,
                        request.max_frames_limit,
                        request.image_width,
                        request.image_height,
                        request.resolution_mode,
                    )
                    obs_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "Observe and report."}] + obs_media,
                    })

                    obs_payload = {
                        "model": config["model_name"],
                        "messages": obs_messages,
                        "max_tokens": int(config.get("observation_max_tokens", 1024)),
                        "temperature": float(config.get("observation_temperature", 0.4)),
                        "top_p": request.top_p,
                        "min_p": request.min_p,
                        "repetition_penalty": request.repetition_penalty,
                        "stream": False,
                    }
                    # Match video FPS handling from main payload
                    if is_video_file(request.media_path) and request.force_fps:
                        obs_payload["mm_processor_kwargs"] = {"fps": request.video_fps}

                    obs_resp = requests.post(config["api_url"], json=obs_payload, timeout=300)
                    obs_resp.raise_for_status()
                    obs_data = obs_resp.json()
                    raw_obs = obs_data["choices"][0]["message"].get("content", "") or ""
                    observation_text = strip_thinking_from_content(raw_obs).strip()

                    # Surface observation to frontend as a distinct event so the UI
                    # can render it in a collapsible block under the response
                    if observation_text:
                        yield f"data: {json.dumps({'observation': observation_text})}\n\n"

                        # Inject as silent context into the system message of pass B
                        obs_context = f"\n\n[VISUAL OBSERVATION — silent context, do not repeat verbatim in your response]\n{observation_text}\n[/VISUAL OBSERVATION]"
                        if api_messages and api_messages[0]["role"] == "system":
                            api_messages[0]["content"] += obs_context
                        else:
                            api_messages.insert(0, {"role": "system", "content": obs_context.strip()})

                        # Optionally drop the media from pass B to save tokens —
                        # observation already extracted what's there
                        if not config.get("observation_include_media_in_pass_b", True):
                            for i in range(len(api_messages) - 1, -1, -1):
                                if api_messages[i]["role"] == "user" and isinstance(api_messages[i]["content"], list):
                                    api_messages[i]["content"] = [
                                        c for c in api_messages[i]["content"] if c.get("type") == "text"
                                    ]
                                    # Collapse back to plain string if only one text block
                                    if len(api_messages[i]["content"]) == 1:
                                        api_messages[i]["content"] = api_messages[i]["content"][0].get("text", "")
                                    break
                except Exception as obs_err:
                    # Pass A failure must never break pass B — surface as warning event
                    yield f"data: {json.dumps({'observation_error': str(obs_err)[:500]})}\n\n"

            # Inject pane context into the existing system message (Qwen3.5 only allows one system msg at index 0)
            if request.pane_context:
                context_text = f"\n\n[WORKSPACE CONTEXT] The user is currently working in the following view:\n{request.pane_context}\nUse this context to provide relevant assistance."
                if api_messages and api_messages[0]["role"] == "system":
                    api_messages[0]["content"] += context_text
                else:
                    api_messages.insert(0, {"role": "system", "content": context_text.strip()})

            # Inject live screen observation (rolling context from /api/observe-frame).
            # Same single-system-message rule as pane_context above.
            if request.live_observation:
                live_text = f"\n\n[LIVE SCREEN — what the user is looking at right now, updated continuously; treat as the present moment, do not repeat verbatim]\n{request.live_observation}\n[/LIVE SCREEN]"
                if api_messages and api_messages[0]["role"] == "system":
                    api_messages[0]["content"] += live_text
                else:
                    api_messages.insert(0, {"role": "system", "content": live_text.strip()})

            # Prepare base API payload - Qwen3.5 params go at top level, not in extra_body
            base_payload = {
                "model": config["model_name"],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                # min_p and repetition_penalty are vLLM extra sampling params. They
                # were previously accepted from the UI and silently dropped here, so
                # both sliders were dead.
                "min_p": request.min_p,
                "repetition_penalty": request.repetition_penalty,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
            }
            if request.enable_thinking:
                base_payload["chat_template_kwargs"] = {"enable_thinking": True}

            if request.include_media and request.media_path and is_video_file(request.media_path) and request.force_fps:
                base_payload["mm_processor_kwargs"] = {
                    "fps": request.video_fps,
                }

            if request.seed >= 0:
                base_payload["seed"] = request.seed

            # --- Tool-use loop (when enabled) ---
            if request.tools_enabled:
                base_payload["tools"] = CHAT_TOOLS
                max_tool_rounds = 10  # safety limit

                # Every round is streamed, including the final one. The old code ran
                # each round with stream=False to find out whether it held tool calls,
                # which meant the final answer landed in the UI as a single block —
                # i.e. turning Tools ON silently disabled token streaming.
                stream_rounds = config.get("stream_tool_rounds", True)
                tool_fmt = request.tool_call_format if request.tool_call_format in TOOL_CALL_FORMATS else "auto"

                if tool_fmt != "server":
                    # tool_choice="none" stands the server's parser down while still
                    # rendering the tool definitions into the prompt via the chat
                    # template, so the model emits its native call markup and we parse
                    # it here. This is required, not merely tidier: with the parser
                    # engaged but mismatched (hermes against Qwen3 XML, say) vLLM
                    # swallows the call while streaming — no tool_calls AND no content —
                    # leaving nothing to recover.
                    base_payload["tool_choice"] = "none"

                for _round in range(max_tool_rounds):
                    if stream_rounds:
                        payload = {**base_payload, "messages": api_messages, "stream": True}
                        round_result = None
                        async for kind, item in _stream_tool_round(
                            config["api_url"], payload, http_request, tool_fmt
                        ):
                            if kind == "delta":
                                yield f"data: {json.dumps({'content': item})}\n\n"
                            else:
                                round_result = item
                        if round_result is None:
                            break  # disconnected mid-round
                        message = {
                            "role": "assistant",
                            "content": round_result["content"],
                        }
                        if round_result["tool_calls"]:
                            message["tool_calls"] = round_result["tool_calls"]
                        reasoning = round_result["reasoning"]
                        finish_reason = round_result["finish_reason"]
                        # Deltas already reached the browser; don't re-emit them below.
                        already_streamed = True
                    else:
                        # Escape hatch for vLLM tool-call parsers that don't support
                        # streaming: set "stream_tool_rounds": false in config.json.
                        payload = {**base_payload, "messages": api_messages, "stream": False}
                        resp = requests.post(config["api_url"], json=payload, timeout=600)
                        resp.raise_for_status()
                        result = resp.json()
                        choice = result["choices"][0]
                        message = choice["message"]
                        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                        finish_reason = choice.get("finish_reason", "")
                        already_streamed = False
                        # Same fallback as the streaming path: recover a tool call the
                        # server's parser left sitting in the content.
                        if not message.get("tool_calls"):
                            recovered = parse_inline_tool_calls(message.get("content") or "", tool_fmt)
                            if recovered:
                                message["tool_calls"] = recovered
                                opener = min(
                                    (i for i in (message["content"].find(o) for o in _TOOL_OPENERS) if i != -1),
                                    default=-1,
                                )
                                if opener >= 0:
                                    message["content"] = message["content"][:opener]

                    # If the model wants to call tools
                    if finish_reason == "tool_calls" or message.get("tool_calls"):
                        # Emit any text content (and reasoning, wrapped) the model produced alongside tool calls
                        if not already_streamed:
                            tool_pre = (f"<think>{reasoning}</think>" if reasoning else "") + normalize_reasoning_channels(message.get("content") or "")
                            if tool_pre:
                                yield f"data: {json.dumps({'content': tool_pre})}\n\n"

                        # Add assistant message (with tool_calls) to conversation
                        api_messages.append(message)

                        for tc in message.get("tool_calls", []):
                            tool_name = tc["function"]["name"]
                            try:
                                tool_args = json.loads(tc["function"]["arguments"])
                            except json.JSONDecodeError:
                                tool_args = {}

                            # Notify frontend about tool execution
                            yield f"data: {json.dumps({'tool_call': {'id': tc['id'], 'name': tool_name, 'arguments': tool_args}})}\n\n"

                            # Execute the tool
                            tool_result = execute_tool(tool_name, tool_args, config)

                            # Notify frontend of result
                            yield f"data: {json.dumps({'tool_result': {'id': tc['id'], 'name': tool_name, 'result': tool_result}})}\n\n"

                            # Add tool result to conversation for next round
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps(tool_result),
                            })

                        continue  # next round — model sees tool results

                    # No tool calls — model produced a final text response.
                    # Re-wrap reasoning so frontend regex and thinking_logs save work.
                    final = (f"<think>{reasoning}</think>" if reasoning else "") + normalize_reasoning_channels(message.get("content") or "")
                    if not already_streamed and final:
                        yield f"data: {json.dumps({'content': final})}\n\n"
                    if final:
                        _save_thinking_from_response(final)
                    yield f"data: [DONE]\n\n"
                    return  # exit generate()

                # Exhausted tool rounds
                yield f"data: {json.dumps({'content': '\\n\\n*[Tool loop limit reached]*'})}\n\n"
                yield f"data: [DONE]\n\n"
                return

            # --- Standard streaming path (no tools) ---
            payload = {**base_payload, "messages": api_messages, "stream": True}

            loop = asyncio.get_event_loop()
            vllm_response = await loop.run_in_executor(
                None,
                lambda: requests.post(config["api_url"], json=payload, stream=True, timeout=600)
            )
            vllm_response.raise_for_status()

            # Feed blocking iter_lines() into an asyncio queue from a background thread
            line_queue: asyncio.Queue = asyncio.Queue()

            def _stream_lines():
                try:
                    for line in vllm_response.iter_lines():
                        loop.call_soon_threadsafe(line_queue.put_nowait, line)
                except Exception:
                    pass
                finally:
                    loop.call_soon_threadsafe(line_queue.put_nowait, None)  # sentinel

            import threading
            threading.Thread(target=_stream_lines, daemon=True).start()

            accumulated = []
            think_open = False  # tracks whether a <think> block is currently open in the stream
            try:
                while True:
                    if await http_request.is_disconnected():
                        break
                    try:
                        line = await asyncio.wait_for(line_queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue
                    if line is None:
                        break
                    if line:
                        decoded = line.decode('utf-8') if isinstance(line, bytes) else line
                        if decoded.startswith('data: '):
                            decoded = decoded[6:]
                        if decoded == '[DONE]':
                            if accumulated:
                                _save_thinking_from_response("".join(accumulated))
                            yield f"data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(decoded)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                # vLLM's qwen3 reasoning parser routes thinking into a separate
                                # `reasoning` field (older variants use `reasoning_content`).
                                # Re-wrap it inline in <think>…</think> so the frontend, the
                                # thinking_logs save path, and the TTS strip all keep working.
                                reasoning = delta.get('reasoning') or delta.get('reasoning_content') or ''
                                content = delta.get('content') or ''
                                if reasoning:
                                    if not think_open:
                                        think_open = True
                                        accumulated.append('<think>')
                                        yield f"data: {json.dumps({'content': '<think>'})}\n\n"
                                    accumulated.append(reasoning)
                                    yield f"data: {json.dumps({'content': reasoning})}\n\n"
                                if content:
                                    if think_open:
                                        think_open = False
                                        accumulated.append('</think>')
                                        yield f"data: {json.dumps({'content': '</think>'})}\n\n"
                                    accumulated.append(content)
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue
            except GeneratorExit:
                pass
            finally:
                vllm_response.close()  # drops TCP connection → vLLM aborts the request

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.delete("/api/media/{media_id}")
async def delete_media(media_id: str):
    """Delete uploaded media"""
    for f in UPLOAD_DIR.iterdir():
        if f.stem == media_id:
            f.unlink()
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Media not found")


# --- Caption Reviewer Endpoints ---

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SaveCaptionRequest(BaseModel):
    caption_path: str
    caption: str


@app.get("/api/captions/scan")
async def scan_captions(directory: str = Query(...)):
    """Scan a directory for image-caption pairs. Returns metadata only, not content."""
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    pairs = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            caption_path = f.with_suffix(".txt")
            has_caption = caption_path.exists()
            caption = ""
            if has_caption:
                caption = caption_path.read_text(encoding="utf-8").strip()
            pairs.append({
                "image_path": str(f),
                "caption_path": str(caption_path),
                "filename": f.name,
                "caption": caption,
                "has_caption": has_caption,
            })

    return {
        "pairs": pairs,
        "total": len(pairs),
        "with_captions": sum(1 for p in pairs if p["has_caption"]),
    }


@app.get("/api/captions/scan-dual")
async def scan_captions_dual(
    directory: str = Query(...),
    subdir_a: str = Query(...),
    subdir_b: str = Query(""),
):
    """Scan a base directory for images; resolve captions from two named subdirectories.

    Images live in ``directory``.  Caption .txt files are expected at
    ``directory/subdir_a/<stem>.txt`` and optionally ``directory/subdir_b/<stem>.txt``.
    The save endpoint already creates parent dirs, so subdirs need not pre-exist.
    """
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    path_a = dir_path / subdir_a
    path_b = dir_path / subdir_b if subdir_b else None

    pairs = []
    for f in sorted(dir_path.iterdir()):
        if not (f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS):
            continue
        stem = f.stem

        cap_a = path_a / f"{stem}.txt"
        has_a = cap_a.exists()
        caption_a = cap_a.read_text(encoding="utf-8").strip() if has_a else ""

        cap_b = (path_b / f"{stem}.txt") if path_b else None
        has_b = cap_b.exists() if cap_b else False
        caption_b = cap_b.read_text(encoding="utf-8").strip() if has_b else ""

        pairs.append({
            "image_path": str(f),
            "filename": f.name,
            "caption_path_a": str(cap_a),
            "caption_a": caption_a,
            "has_caption_a": has_a,
            "caption_path_b": str(cap_b) if cap_b else "",
            "caption_b": caption_b,
            "has_caption_b": has_b,
        })

    return {
        "pairs": pairs,
        "total": len(pairs),
        "with_captions_a": sum(1 for p in pairs if p["has_caption_a"]),
        "with_captions_b": sum(1 for p in pairs if p["has_caption_b"]),
    }


@app.get("/api/captions/batch-scan")
async def batch_scan_captions(directory: str = Query(...)):
    """Recursively scan subdirectories for images, grouped by subdirectory."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    # Collect all dirs that contain images (root + all subdirs)
    all_dirs = sorted([root] + [d for d in root.rglob("*") if d.is_dir()])
    subdirs = []
    for d in all_dirs:
        pairs = []
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                caption_path = f.with_suffix(".txt")
                pairs.append({
                    "image_path": str(f),
                    "caption_path": str(caption_path),
                    "filename": f.name,
                    "has_caption": caption_path.exists(),
                })
        if pairs:
            rel = str(d.relative_to(root)) if d != root else "."
            subdirs.append({
                "dir": str(d),
                "rel_dir": rel,
                "images": pairs,
                "total": len(pairs),
            })

    return {
        "subdirs": subdirs,
        "total_subdirs": len(subdirs),
        "total_images": sum(s["total"] for s in subdirs),
    }


@app.get("/api/captions/image")
async def get_caption_image(path: str = Query(...)):
    """Serve an image file for the caption reviewer."""
    file_path = Path(path)
    if not file_path.exists() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image not found")

    mime = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    return StreamingResponse(open(file_path, "rb"), media_type=mime)


@app.post("/api/captions/save")
async def save_caption(request: SaveCaptionRequest):
    """Save an edited caption back to its .txt file."""
    caption_path = Path(request.caption_path)
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(request.caption.strip(), encoding="utf-8")
    return {"status": "saved", "path": str(caption_path)}


class RecaptionRequest(BaseModel):
    image_path: str
    existing_caption: str = ""
    extra_instruction: str = ""
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    enable_thinking: bool = False
    strip_thinking: bool = True


class RotateImageRequest(BaseModel):
    image_path: str
    degrees: int  # positive = CCW (PIL convention): 90=CCW, -90=CW, 180=flip


@app.post("/api/captions/rotate")
async def rotate_image(request: RotateImageRequest):
    """Rotate an image in-place and save it back to disk."""
    path = Path(request.image_path)
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Image not found")

    with Image.open(path) as img:
        fmt = img.format or path.suffix.lstrip(".").upper()
        if fmt == "JPG":
            fmt = "JPEG"
        # Apply EXIF orientation first so rotations are visually correct
        img = ImageOps.exif_transpose(img)
        img.load()  # force full read into memory before overwriting the file
        rotated = img.rotate(request.degrees, expand=True)

    save_kwargs: dict = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0

    rotated.save(path, **save_kwargs)
    return {"status": "rotated", "path": str(path)}


class DeleteCaptionRequest(BaseModel):
    caption_path: str
    image_path: str


@app.delete("/api/captions/delete")
async def delete_caption(request: DeleteCaptionRequest):
    """Delete a caption .txt file and its associated image."""
    deleted = []
    for path_str in (request.caption_path, request.image_path):
        p = Path(path_str)
        if p.exists():
            p.unlink()
            deleted.append(path_str)
    return {"status": "deleted", "deleted": deleted}


@app.post("/api/captions/rerun")
async def rerun_caption(req: RecaptionRequest):
    """Re-caption a single image using the current model, optionally with existing caption context
    and extra instructions. Streams the result as SSE."""
    config = load_config()

    def generate():
        image_path = req.image_path
        file_path = Path(image_path)
        if not file_path.exists():
            yield f"data: {json.dumps({'error': f'File not found: {image_path}'})}\n\n"
            return

        parts = []
        if req.existing_caption.strip():
            parts.append(f"Existing caption:\n{req.existing_caption.strip()}")
        if req.extra_instruction.strip():
            parts.append(req.extra_instruction.strip())
        else:
            parts.append("Review the image and write an improved, accurate caption.")
        instruction = "\n\n".join(parts)

        content = [
            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
            {"type": "text", "text": instruction},
        ]
        payload = {
            "model": config["model_name"],
            "messages": [{"role": "user", "content": content}],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "min_p": req.min_p,
            "top_k": req.top_k,
            "repetition_penalty": req.repetition_penalty,
            "presence_penalty": req.presence_penalty,
            "frequency_penalty": req.frequency_penalty,
            "stream": True,
        }
        if req.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

        try:
            response = requests.post(config["api_url"], json=payload, stream=True, timeout=600)
            response.raise_for_status()
            accumulated = []
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    if decoded == "[DONE]":
                        full = "".join(accumulated)
                        if req.strip_thinking:
                            full, thinking = _strip_thinking_tags(full)
                            if thinking:
                                image_dir = file_path.parent
                                thinking_dir = image_dir / "thinking_text"
                                thinking_dir.mkdir(parents=True, exist_ok=True)
                                (thinking_dir / f"{file_path.stem}_thinking.txt").write_text(thinking, encoding="utf-8")
                        yield f"data: {json.dumps({'done': True, 'caption': full})}\n\n"
                        break
                    try:
                        chunk = json.loads(decoded)
                        token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token:
                            accumulated.append(token)
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- Lazy media captioning (durable "visual memory" for chat history) ---

# Neutral, factual summarization wrapper. Defined in source (never config.json)
# so a stale config can't inject a prompt here. The active mode's media prompt
# (modes.yaml) is prepended as interpretation framing when requested.
MEDIA_MEMORY_INSTRUCTION = (
    "You are writing a durable visual memory note. The media will soon leave the "
    "conversation, and this summary will be the ONLY record that remains of it. "
    "Write a concise, factual description (3-5 sentences) capturing the key visible "
    "elements: subjects, setting, actions, and any notable detail later turns might "
    "reference. Describe only what is visible; do not speculate or answer questions. "
    "Output only the summary text."
)


class CaptionMediaRequest(BaseModel):
    media_id: str
    interaction_mode: str = "Free-form"
    active_character: str = ""
    include_mode_framing: bool = True
    # Media processing params (mirror the chat request so the caption sees the
    # same representation the model originally saw).
    processing_mode: str = ""
    sampling_mode: str = ""
    interval: float = 0
    target_fps: float = 0
    max_frames_limit: int = 0
    image_width: int = 0
    image_height: int = 0
    resolution_mode: str = ""
    video_fps: float = 0
    force_fps: bool = False
    max_tokens: int = 0


@app.post("/api/caption-media")
async def caption_media(req: CaptionMediaRequest):
    """Generate a durable text caption for an uploaded media item (by id).

    Used to collapse media to a 'visual memory' note once it falls out of the
    image window in chat history. Non-streaming: returns {"caption": str}.
    """
    config = load_config()

    # Resolve media_id -> file in uploads/ (same lookup as /api/media/{id})
    file_path: Optional[Path] = None
    for f in UPLOAD_DIR.iterdir():
        if f.stem == req.media_id:
            file_path = f
            break
    if file_path is None:
        return {"error": f"Media not found: {req.media_id}"}

    media_path = str(file_path)
    is_video = is_video_file(media_path)

    # Interpretation framing from the active mode (modes.yaml), never config.json.
    instruction = MEDIA_MEMORY_INSTRUCTION
    if req.include_mode_framing:
        mode_bundle = resolve_ui_mode(req.interaction_mode, req.active_character)
        framing = mode_bundle["media_video"] if is_video else mode_bundle["media_image"]
        if framing:
            instruction = f"{framing}\n\n{MEDIA_MEMORY_INSTRUCTION}"

    try:
        media_content = prepare_media_content(
            media_path,
            req.processing_mode,
            req.sampling_mode,
            req.interval,
            req.target_fps,
            req.max_frames_limit,
            req.image_width,
            req.image_height,
            req.resolution_mode,
        )
        if not media_content:
            return {"error": "Could not prepare media content"}

        payload = {
            "model": config["model_name"],
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": [{"type": "text", "text": "Summarize this media."}] + media_content},
            ],
            "max_tokens": int(req.max_tokens or config.get("observation_max_tokens", 1024)),
            "temperature": float(config.get("observation_temperature", 0.4)),
            "stream": False,
        }
        if is_video and req.force_fps:
            payload["mm_processor_kwargs"] = {"fps": req.video_fps}

        resp = requests.post(config["api_url"], json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"].get("content", "") or ""
        caption = strip_thinking_from_content(raw).strip()
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)[:500]}


# --- Chat Log Endpoints ---

class SaveChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    title: Optional[str] = None
    sampling: Optional[Dict[str, Any]] = None


@app.post("/api/chat/save")
async def save_chat_log(request: SaveChatRequest):
    """Save a conversation to disk."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    title = request.title or "chat"
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)[:60].strip()
    filename = f"{ts}_{safe_title}.json".replace(" ", "_")
    filepath = CHAT_LOGS_DIR / filename

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "title": request.title,
        "message_count": len(request.messages),
        "messages": request.messages,
    }
    if request.sampling:
        log_data["sampling"] = request.sampling
    filepath.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "saved", "path": str(filepath), "filename": filename}


@app.get("/api/chat/logs")
async def list_chat_logs():
    """List saved conversation logs."""
    logs = []
    for f in sorted(CHAT_LOGS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            logs.append({
                "filename": f.name,
                "timestamp": data.get("timestamp"),
                "title": data.get("title"),
                "message_count": data.get("message_count", 0),
            })
        except Exception:
            continue
    return {"logs": logs}


def _safe_chat_log_path(filename: str) -> Path:
    """Resolve a chat-log filename to a path inside CHAT_LOGS_DIR (no traversal)."""
    candidate = (CHAT_LOGS_DIR / filename).resolve()
    if candidate.parent != CHAT_LOGS_DIR.resolve() or candidate.suffix != ".json":
        raise HTTPException(status_code=400, detail="Invalid log filename")
    return candidate


@app.get("/api/chat/logs/{filename}")
async def load_chat_log(filename: str):
    """Load a saved conversation (messages + metadata) for reloading into the UI."""
    filepath = _safe_chat_log_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/logs/{filename}")
async def delete_chat_log(filename: str):
    """Delete a saved conversation log."""
    filepath = _safe_chat_log_path(filename)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    filepath.unlink()
    return {"status": "deleted", "filename": filename}


# --- Batch Captioning ---

_batch_stop_flag = False
_batch_running = False

MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm"}
VIDEO_EXTENSIONS_SET = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_IMAGE_INSTRUCTION = (
    "Analyze this image and provide a detailed caption in the following EXACT format. Fill in ALL sections:\n\n"
    "[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>\n"
    "[TEXT]: <Any on-screen text visible. If none, write \"None\">\n\n"
    "You MUST fill in both sections."
)

DEFAULT_VIDEO_INSTRUCTION = (
    "Analyze this video and provide a detailed caption in the following EXACT format. Fill in ALL sections:\n\n"
    "[VISUAL]: <Detailed description of people, objects, actions, settings, colors, movements, and scene transitions>\n"
    "[SPEECH]: <Word-for-word transcription of everything spoken. If no speech, write \"None\">\n"
    "[SOUNDS]: <Description of music, ambient sounds, sound effects. If none, write \"None\">\n"
    "[TEXT]: <Any on-screen text visible. If none, write \"None\">\n\n"
    "You MUST fill in all four sections."
)


def _strip_thinking_tags(text: str) -> tuple:
    """Remove <think>...</think> blocks from model output.

    Returns (stripped_text, thinking_text) tuple.
    """
    import re
    if not isinstance(text, str):
        return text, ""
    lower = text.lower()
    idx = lower.find("</think>")
    if idx != -1:
        thinking = text[:idx + len("</think>")]
        stripped = text[idx + len("</think>"):].strip()
        return stripped, thinking
    match = re.search(r"<think>.*?</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        thinking = match.group(0)
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        return stripped, thinking
    return text, ""


def _resolve_caption_instruction(is_video: bool, instruction: str, caption_target: str) -> tuple[str, int]:
    """Resolve the captioning instruction and max_tokens from target or fallback.

    Returns (instruction, max_tokens_override_or_0).
    """
    prompts = load_prompts()
    resolved = None
    tokens = 0

    if caption_target:
        target = prompts.get("caption_targets", {}).get(caption_target, {})
        if target:
            key = "video_instruction" if is_video else "image_instruction"
            target_instruction = target.get(key)
            if target_instruction:
                resolved = target_instruction
                if instruction:
                    resolved += f"\n\nAdditional instructions:\n{instruction}"
                tokens = target.get("max_tokens", 0)

    if resolved is None:
        if not instruction:
            batch_cfg = prompts.get("batch_captioner", {})
            if is_video:
                # `or` (not .get's default) so a present-but-empty "" in prompts.json
                # still falls back to the built-in default instead of an empty prompt.
                instruction = batch_cfg.get("video_instruction") or DEFAULT_VIDEO_INSTRUCTION
            else:
                instruction = batch_cfg.get("image_instruction") or DEFAULT_IMAGE_INSTRUCTION
        resolved = instruction

    return resolved, tokens


def _caption_single_file(file_path: str, instruction: str, config: dict,                          req: 'BatchCaptionRequest') -> str:
    """Send a single media file to vLLM for captioning."""

    """
    Generate a caption for a single media file using vLLM API.
    This function sends a media file (image or video) to a vLLM service for captioning.
    It constructs an appropriate payload based on the media type and sends it via HTTP POST request.
    Args:
        file_path (str): The full file path including the file extension (e.g., '/path/to/image.jpg' 
                            or '/path/to/video.mp4'). This is not just the base directory, but the 
                            complete path to the specific media file to be captioned.
        instruction (str): The captioning instruction or prompt to send to the model.
        config (dict): Configuration dictionary containing 'model_name' and 'api_url' keys for 
                        the vLLM service.
        req (BatchCaptionRequest): Request object containing captioning parameters including 
                                    max_tokens, temperature, top_p, top_k, presence_penalty, 
                                    enable_thinking, strip_thinking, caption_target, and video_fps.
    Returns:
        str: The generated caption text from the model, with thinking tags stripped if 
                req.strip_thinking is True.
    Raises:
        requests.HTTPError: If the API request fails (via response.raise_for_status()).
        requests.Timeout: If the request exceeds 600 seconds timeout.
        KeyError: If required keys are missing from config or response JSON.
    """

    media_type = get_media_type(file_path)
    is_video = media_type == "video"

    instruction, tokens_override = _resolve_caption_instruction(
        is_video, instruction, req.caption_target
    )
    if tokens_override:
        req_max_tokens = tokens_override
    else:
        req_max_tokens = req.max_tokens

    if is_video:
        content = [
            {"type": "video_url", "video_url": {"url": f"file://{file_path}"}},
            {"type": "text", "text": instruction},
        ]
    else:
        content = [
            {"type": "image_url", "image_url": {"url": f"file://{file_path}"}},
            {"type": "text", "text": instruction},
        ]

    messages = [{"role": "user", "content": content}]

    # Assistant-prefill: seed the reply so the model continues in the intended
    # explicit register instead of opening with a euphemism/hedge. Per-request
    # value wins; else fall back to prompts.json batch_captioner.prefill.
    prefill = req.caption_prefill
    if prefill is None:
        prefill = load_prompts().get("batch_captioner", {}).get("prefill", "") or ""
    prefill = prefill.strip()

    payload = {
        "model": config["model_name"],
        "messages": messages,
        "max_tokens": req_max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "min_p": req.min_p,
        "top_k": req.top_k,
        "repetition_penalty": req.repetition_penalty,
        "presence_penalty": req.presence_penalty,
        "frequency_penalty": req.frequency_penalty,
        "stream": False,
    }
    if req.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    if is_video and req.force_fps:
        payload["mm_processor_kwargs"] = {
            "fps": req.video_fps,
        }

    if prefill:
        # Append the seed as a partial assistant turn and have vLLM continue it
        # rather than start a fresh (hedge-prone) reply. The API returns only the
        # continuation, so we prepend the seed back to rebuild the full caption.
        messages.append({"role": "assistant", "content": prefill})
        payload["add_generation_prompt"] = False
        payload["continue_final_message"] = True

    response = requests.post(config["api_url"], json=payload, timeout=600)
    response.raise_for_status()
    continuation = response.json()["choices"][0]["message"]["content"]
    result = ((prefill + continuation) if prefill else continuation).strip()

    if req.strip_thinking:
        result, thinking = _strip_thinking_tags(result)
        if thinking:
            thinking_dir = Path(req.directory) / "thinking_text"
            thinking_dir.mkdir(parents=True, exist_ok=True)
            thinking_file = thinking_dir / f"{Path(file_path).stem}_thinking.txt"
            thinking_file.write_text(thinking, encoding="utf-8")

    return result


@app.post("/api/batch/stop")
async def batch_stop():
    global _batch_stop_flag
    _batch_stop_flag = True
    return {"status": "stopping"}


@app.get("/api/batch/status")
async def batch_status():
    return {"running": _batch_running}


@app.post("/api/batch/caption")
async def batch_caption(req: BatchCaptionRequest):
    """Stream batch captioning progress via SSE."""
    global _batch_stop_flag, _batch_running
    _batch_stop_flag = False
    _batch_running = True

    config = load_config()
    directory = Path(req.directory).expanduser().resolve()

    async def generate():
        global _batch_stop_flag, _batch_running
        try:
            if not directory.is_dir():
                yield f"data: {json.dumps({'error': f'Directory not found: {req.directory}'})}\n\n"
                return

            media_files = sorted(
                f for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
            )

            if not media_files:
                yield f"data: {json.dumps({'error': 'No media files found'})}\n\n"
                return

            output_path = directory / "captions.json"
            existing = {}
            if req.skip_existing and output_path.exists():
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        for entry in json.load(f):
                            existing[entry["file"]] = entry
                except Exception:
                    pass

            results = dict(existing)
            total = len(media_files)
            completed = 0
            skipped = 0
            errors = 0

            yield f"data: {json.dumps({'type': 'start', 'total': total, 'existing': len(existing)})}\n\n"

            for i, media_file in enumerate(media_files):
                if _batch_stop_flag:
                    yield f"data: {json.dumps({'type': 'stopped', 'completed': completed, 'total': total})}\n\n"
                    break

                file_name = media_file.name

                if req.skip_existing and file_name in existing:
                    skipped += 1
                    yield f"data: {json.dumps({'type': 'skip', 'file': file_name, 'index': i, 'total': total})}\n\n"
                    continue

                yield f"data: {json.dumps({'type': 'processing', 'file': file_name, 'index': i, 'total': total})}\n\n"

                try:
                    caption = await asyncio.to_thread(
                        _caption_single_file,
                        str(media_file), req.instruction, config, req
                    )
                    results[file_name] = {"file": file_name, "caption": caption}
                    completed += 1

                    tmp = output_path.with_suffix(".tmp")
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(list(results.values()), f, indent=2, ensure_ascii=False)
                    tmp.rename(output_path)

                    yield f"data: {json.dumps({'type': 'done', 'file': file_name, 'index': i, 'total': total, 'completed': completed, 'caption_preview': caption[:200]})}\n\n"

                except Exception as e:
                    errors += 1
                    yield f"data: {json.dumps({'type': 'error', 'file': file_name, 'index': i, 'error': str(e)})}\n\n"

                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'type': 'complete', 'completed': completed, 'skipped': skipped, 'errors': errors, 'total': total, 'output': str(output_path)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            _batch_running = False

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

class BatchRecaptionRequest(BaseModel):
    directory: str
    extra_instruction: str = ""
    skip_missing: bool = True       # skip images that have no existing .txt caption
    overwrite: bool = True          # overwrite existing .txt files (always True for a re-pass)
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    enable_thinking: bool = False
    strip_thinking: bool = True


_batch_recaption_stop_flag = False
_batch_recaption_running = False


@app.post("/api/batch/recaption/stop")
async def batch_recaption_stop():
    global _batch_recaption_stop_flag
    _batch_recaption_stop_flag = True
    return {"status": "stopping"}


@app.get("/api/batch/recaption/status")
async def batch_recaption_status():
    return {"running": _batch_recaption_running}


@app.post("/api/batch/recaption")
async def batch_recaption(req: BatchRecaptionRequest):
    """Stream a batch re-caption pass over all images in a directory.
    Sends each image + its existing .txt caption to the model with optional extra instructions.
    Overwrites .txt files with the new result."""
    global _batch_recaption_stop_flag, _batch_recaption_running
    _batch_recaption_stop_flag = False
    _batch_recaption_running = True

    config = load_config()
    directory = Path(req.directory).expanduser().resolve()

    async def generate():
        global _batch_recaption_stop_flag, _batch_recaption_running
        try:
            if not directory.is_dir():
                yield f"data: {json.dumps({'error': f'Directory not found: {req.directory}'})}\n\n"
                return

            image_files = sorted(f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            if not image_files:
                yield f"data: {json.dumps({'error': 'No image files found'})}\n\n"
                return

            # Filter to only images with captions if skip_missing
            if req.skip_missing:
                image_files = [f for f in image_files if f.with_suffix(".txt").exists()]
                if not image_files:
                    yield f"data: {json.dumps({'error': 'No captioned images found (all missing .txt files)'})}\n\n"
                    return

            total = len(image_files)
            completed = 0
            skipped = 0
            errors = 0

            yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

            for i, image_file in enumerate(image_files):
                if _batch_recaption_stop_flag:
                    yield f"data: {json.dumps({'type': 'stopped', 'completed': completed, 'total': total})}\n\n"
                    break

                caption_path = image_file.with_suffix(".txt")
                existing_caption = ""
                if caption_path.exists():
                    existing_caption = caption_path.read_text(encoding="utf-8").strip()
                elif req.skip_missing:
                    skipped += 1
                    yield f"data: {json.dumps({'type': 'skip', 'file': image_file.name, 'index': i, 'total': total})}\n\n"
                    continue

                yield f"data: {json.dumps({'type': 'processing', 'file': image_file.name, 'index': i, 'total': total})}\n\n"

                try:
                    parts = []
                    if existing_caption:
                        parts.append(f"Existing caption:\n{existing_caption}")
                    if req.extra_instruction.strip():
                        parts.append(req.extra_instruction.strip())
                    else:
                        parts.append("Review the image and write an improved, accurate caption.")
                    instruction = "\n\n".join(parts)

                    content = [
                        {"type": "image_url", "image_url": {"url": f"file://{image_file}"}},
                        {"type": "text", "text": instruction},
                    ]
                    payload = {
                        "model": config["model_name"],
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature,
                        "top_p": req.top_p,
                        "min_p": req.min_p,
                        "top_k": req.top_k,
                        "repetition_penalty": req.repetition_penalty,
                        "presence_penalty": req.presence_penalty,
                        "frequency_penalty": req.frequency_penalty,
                        "stream": False,
                    }
                    if req.enable_thinking:
                        payload["chat_template_kwargs"] = {"enable_thinking": True}

                    def _call():
                        r = requests.post(config["api_url"], json=payload, timeout=600)
                        r.raise_for_status()
                        return r.json()["choices"][0]["message"]["content"].strip()

                    result = await asyncio.to_thread(_call)

                    if req.strip_thinking:
                        result, thinking = _strip_thinking_tags(result)
                        if thinking:
                            thinking_dir = directory / "thinking_text"
                            thinking_dir.mkdir(parents=True, exist_ok=True)
                            (thinking_dir / f"{image_file.stem}_thinking.txt").write_text(thinking, encoding="utf-8")

                    caption_path.write_text(result, encoding="utf-8")
                    completed += 1
                    yield f"data: {json.dumps({'type': 'done', 'file': image_file.name, 'index': i, 'total': total, 'completed': completed, 'caption_preview': result[:200]})}\n\n"

                except Exception as e:
                    errors += 1
                    yield f"data: {json.dumps({'type': 'error', 'file': image_file.name, 'index': i, 'error': str(e)})}\n\n"

                await asyncio.sleep(0.1)

            yield f"data: {json.dumps({'type': 'complete', 'completed': completed, 'skipped': skipped, 'errors': errors, 'total': total})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            _batch_recaption_running = False

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
