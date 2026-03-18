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
import tempfile
import subprocess
import mimetypes
import asyncio
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
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
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CHAT_LOGS_DIR = Path(__file__).parent / "chat_logs"
CHAT_LOGS_DIR.mkdir(exist_ok=True)
THINKING_LOGS_DIR = Path(__file__).parent / "thinking_logs"
THINKING_LOGS_DIR.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model_name": "Qwen35-9B",
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
    "presence_penalty": 1.5,
    "frequency_penalty": 0.0,
    "seed": -1,
    "thought_syntax": "<think>{content}</think>",
    "vram_limit": 170000
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


def strip_thinking_from_content(content: str) -> str:
    """Remove <think>...</think> blocks from assistant messages for prefix cache efficiency.

    Only the final response is retained in conversation history.
    This saves significant context tokens during multi-turn conversations.
    """
    if not isinstance(content, str):
        return content
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
    }
]

# Allowed base directories for tool file operations (safety constraint)
TOOL_ALLOWED_PATHS = [
    Path("/media/ajax/AI"),
    Path("/home/ajax"),
]


def is_path_allowed(file_path: str) -> bool:
    """Check if a file path is within allowed directories."""
    resolved = Path(file_path).resolve()
    return any(resolved.is_relative_to(base) for base in TOOL_ALLOWED_PATHS)


def execute_tool(name: str, arguments: dict) -> dict:
    """Execute a tool call and return the result."""
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

    return {"error": f"Unknown tool: {name}"}


CUSTOM_INSTRUCTIONS = ""


# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be string or list for multimodal


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
    presence_penalty: float = 1.5
    frequency_penalty: float = 0.0
    seed: int = -1
    # Qwen3.5 thinking mode
    enable_thinking: bool = False
    # Mode settings
    interaction_mode: str = "Free-form"
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
    # Agentic tool use
    tools_enabled: bool = False


class BatchCaptionRequest(BaseModel):
    directory: str
    instruction: str = ""
    caption_target: str = ""
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    enable_thinking: bool = False
    video_fps: float = 2.0
    force_fps: bool = True
    strip_thinking: bool = True
    skip_existing: bool = True
    output_format: str = "json"


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
        if custom_mode:
            return CUSTOM_INSTRUCTIONS.strip()
        return None
    
    thinking_instruction = ""
    if inject_thinking and mode_config["inject_thinking"] and thought_syntax and "{content}" in thought_syntax:
        open_tag = thought_syntax.split("{content}")[0]
        close_tag = thought_syntax.split("{content}")[1]
        thinking_instruction = f"\n\nUse {open_tag} and {close_tag} tags for your internal reasoning before responding."
    
    custom_prefix = CUSTOM_INSTRUCTIONS if custom_mode else ""
    
    if interaction_mode == "Roleplay":
        if system_prompt:
            return custom_prefix + system_prompt + thinking_instruction
        elif custom_mode:
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


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    """Streaming chat endpoint"""
    config = load_config()
    prompts = load_prompts()
    
    async def generate():
        try:
            # Build messages for API
            api_messages = []
            
            # Determine system prompt based on interaction mode:
            # - Free-form: use chat_assistant.system_prompt directly
            # - Roleplay:  use roleplay.system_prompt (falls back to request.system_prompt, then chat_assistant)
            # - Analytical: use request.system_prompt or chat_assistant fallback
            chat_system = prompts.get("chat_assistant", {}).get("system_prompt", "")
            roleplay_system = prompts.get("roleplay", {}).get("system_prompt", "")

            if not request.messages or request.messages[0].role != "system":
                if chat_system and request.interaction_mode == "Free-form":
                    api_messages.append({"role": "system", "content": chat_system})
                else:
                    if request.interaction_mode == "Roleplay":
                        resolved_prompt = roleplay_system or request.system_prompt or chat_system
                    else:
                        resolved_prompt = request.system_prompt or chat_system
                    system_msg = build_system_message(
                        request.interaction_mode,
                        resolved_prompt,
                        request.thought_syntax,
                        request.inject_thinking,
                        request.custom_mode
                    )
                    if system_msg:
                        api_messages.append({"role": "system", "content": system_msg})
            
            # Process messages, stripping thinking blocks from assistant history
            for msg in request.messages:
                content = msg.content
                if msg.role == "assistant" and isinstance(content, str):
                    content = strip_thinking_from_content(content)
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
            
            # Inject pane context into the existing system message (Qwen3.5 only allows one system msg at index 0)
            if request.pane_context:
                context_text = f"\n\n[WORKSPACE CONTEXT] The user is currently working in the following view:\n{request.pane_context}\nUse this context to provide relevant assistance."
                if api_messages and api_messages[0]["role"] == "system":
                    api_messages[0]["content"] += context_text
                else:
                    api_messages.insert(0, {"role": "system", "content": context_text.strip()})

            # Prepare base API payload - Qwen3.5 params go at top level, not in extra_body
            base_payload = {
                "model": config["model_name"],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
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

                for _round in range(max_tool_rounds):
                    payload = {**base_payload, "messages": api_messages, "stream": False}
                    resp = requests.post(config["api_url"], json=payload, timeout=600)
                    resp.raise_for_status()
                    result = resp.json()

                    choice = result["choices"][0]
                    message = choice["message"]
                    finish_reason = choice.get("finish_reason", "")

                    # If the model wants to call tools
                    if finish_reason == "tool_calls" or message.get("tool_calls"):
                        # Emit any text content the model produced alongside tool calls
                        if message.get("content"):
                            yield f"data: {json.dumps({'content': message['content']})}\n\n"

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
                            tool_result = execute_tool(tool_name, tool_args)

                            # Notify frontend of result
                            yield f"data: {json.dumps({'tool_result': {'id': tc['id'], 'name': tool_name, 'result': tool_result}})}\n\n"

                            # Add tool result to conversation for next round
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps(tool_result),
                            })

                        continue  # next round — model sees tool results

                    # No tool calls — model produced a final text response
                    if message.get("content"):
                        yield f"data: {json.dumps({'content': message['content']})}\n\n"
                        _save_thinking_from_response(message["content"])
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
                                content = chunk['choices'][0].get('delta', {}).get('content', '')
                                if content:
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
    top_k: int = 20
    presence_penalty: float = 1.5
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
            "top_k": req.top_k,
            "presence_penalty": req.presence_penalty,
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


# --- Chat Log Endpoints ---

class SaveChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    title: Optional[str] = None


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

    if caption_target:
        targets = prompts.get("caption_targets", {})
        target = targets.get(caption_target, {})
        if target:
            key = "video_instruction" if is_video else "image_instruction"
            target_instruction = target.get(key)
            if target_instruction:
                final = target_instruction
                if instruction:
                    final += f"\n\nAdditional instructions:\n{instruction}"
                return final, target.get("max_tokens", 0)

    if not instruction:
        batch_cfg = prompts.get("batch_captioner", {})
        if is_video:
            instruction = batch_cfg.get("video_instruction", DEFAULT_VIDEO_INSTRUCTION)
        else:
            instruction = batch_cfg.get("image_instruction", DEFAULT_IMAGE_INSTRUCTION)

    return instruction, 0


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

    payload = {
        "model": config["model_name"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": req_max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "presence_penalty": req.presence_penalty,
        "stream": False,
    }
    if req.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    if is_video and req.force_fps:
        payload["mm_processor_kwargs"] = {
            "fps": req.video_fps,
        }

    response = requests.post(config["api_url"], json=payload, timeout=600)
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"].strip()

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
    top_k: int = 20
    presence_penalty: float = 1.5
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
                        "top_k": req.top_k,
                        "presence_penalty": req.presence_penalty,
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
