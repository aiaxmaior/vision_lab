#!/usr/bin/env python3
"""
Gradio UI for image captioning with koboldcpp / OpenAI-compatible backends.

Designed for large VL models (e.g. Qwen3.5-VL-35B-A3B) running under tight
VRAM / context budgets.  Batch mode processes images **one at a time** in a
loop, writing each result to disk immediately so nothing is lost if the
backend crashes mid-run.

Usage:
    python scripts/caption_ui.py
    python scripts/caption_ui.py --port 7860 --server-port 5001
"""

import gc
import json
import re
import time
from pathlib import Path

import gradio as gr

try:
    from prompt_loader import PromptLoader as _PromptLoader
except ImportError:
    _PromptLoader = None

# ============================================================================
# Caption Processing Utilities
# ============================================================================


def format_thinking_caption(caption: str) -> tuple[str, str]:
    """Parse thinking model output into reasoning and final caption.

    Qwen3-VL frequently omits the opening ``<think>`` tag but always
    emits ``</think>``.  We therefore treat *everything* before the
    first ``</think>`` as reasoning, regardless of whether an opening
    tag is present.  Matching is case-insensitive.
    """
    # Case-insensitive search for the closing tag
    lower = caption.lower()
    idx = lower.find("</think>")
    if idx != -1:
        reasoning_raw = caption[:idx]
        final_caption = caption[idx + len("</think>"):].strip()
        # Strip any opening <think> tag that may or may not be present
        clean = re.sub(r"<think>", "", reasoning_raw, flags=re.IGNORECASE).strip()
        return clean, final_caption

    return "", caption.strip()


def format_caption_as_markdown(media_name: str, caption: str) -> str:
    """Format a single caption as markdown."""
    reasoning, final_caption = format_thinking_caption(caption)

    lines = [f"## {media_name}\n"]

    if reasoning:
        sentences = re.split(r"(?<=[.!?])\s+", reasoning)
        sentences = [s.strip() for s in sentences if s.strip()]
        lines.append("### Reasoning\n")
        lines.append("\n".join(sentences))
        lines.append("")

    lines.append("### Final Caption\n")

    section_pattern = r"\[([A-Z]+)\]:\s*"
    parts = re.split(f"({section_pattern})", final_caption)

    formatted_sections = []
    i = 0
    while i < len(parts):
        part = parts[i].strip() if parts[i] else ""
        if re.match(r"\[[A-Z]+\]:", part):
            section_header = part
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            formatted_sections.append(f"**{section_header}** {content}")
            i += 2
        elif part:
            formatted_sections.append(part)
            i += 1
        else:
            i += 1

    lines.append("\n\n".join(formatted_sections))

    return "\n".join(lines)


def extract_final_caption(caption: str) -> str:
    """Extract only the final caption (strip thinking/reasoning)."""
    _, final_caption = format_thinking_caption(caption)
    return final_caption


# ============================================================================
# Captioner backend (OpenAI-compatible, works with koboldcpp / vLLM / etc.)
# ============================================================================


class Captioner:
    """Sends one media file at a time to an OpenAI-compatible chat/completions endpoint.

    Supports two backends:
      - **vLLM** (default): Uses the OpenAI-compatible API with Qwen3.5 extensions
        including video_url input, thinking mode control via chat_template_kwargs,
        and local file path passthrough (requires --allowed-local-media-path /).
      - **koboldcpp**: Legacy mode with image-only support, downscaling, KV cache
        management, and explicit stop tokens.
    """

    IMAGE_INSTRUCTION = """\
Analyze this image and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in both sections."""

    VIDEO_INSTRUCTION = """\
Analyze this video and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, movements, and scene transitions>
[SPEECH]: <Word-for-word transcription of everything spoken. If no speech, write "None">
[SOUNDS]: <Description of music, ambient sounds, sound effects. If none, write "None">
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in all four sections. For [SPEECH], transcribe the actual words spoken, not a summary."""

    DEFAULT_INSTRUCTION = IMAGE_INSTRUCTION

    def __init__(
        self,
        server_url: str = "http://localhost",
        port: int = 8000,
        model_name: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        max_image_size: int = 768,
        enable_thinking: bool = True,
        backend: str = "vllm",
        video_fps: float = 2.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.port = port
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.max_image_size = max_image_size
        self.enable_thinking = enable_thinking
        self.backend = backend
        self.video_fps = video_fps
        self._base_url = f"{self.server_url}:{self.port}"

    def _detect_model(self) -> str:
        """Auto-detect model name from server."""
        import requests

        try:
            response = requests.get(f"{self._base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            if models:
                return models[0]["id"]
        except Exception:
            pass
        return "default"

    def get_model_name(self) -> str:
        if self.model_name is None:
            self.model_name = self._detect_model()
        return self.model_name

    def _prepare_image_base64(self, image_path: str) -> tuple[str, str]:
        """Load, downscale, and base64-encode an image.

        Returns (base64_string, mime_type).  The image is resized so its
        longest side is at most ``self.max_image_size`` pixels, then saved
        as JPEG (quality 85) to keep the payload small.
        """
        import base64
        from io import BytesIO

        from PIL import Image

        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            w, h = img.size
            longest = max(w, h)
            if longest > self.max_image_size:
                scale = self.max_image_size / longest
                img = img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.LANCZOS,
                )

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)

        b64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        return b64, "image/jpeg"

    @staticmethod
    def sanitize_instruction(instruction: str) -> str:
        """Strip literal <think> / </think> tags and instruction blocks.

        The Qwen3 tokenizer maps these to *special* token IDs rather than
        treating them as text.  When they appear inside the user message,
        they corrupt the generation context and cause koboldcpp to fire
        EOS (token 248046) as soon as the model tries to close its own
        thinking block — aborting generation before the caption is produced.

        Qwen3 thinking models emit <think>…</think> on their own; explicit
        instructions to do so are unnecessary and harmful.
        """
        # Remove <think-instructions>…</think-instructions> wrapper blocks
        cleaned = re.sub(
            r"<think-instructions>.*?</think-instructions>",
            "",
            instruction,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Remove any remaining literal <think> or </think> tags
        cleaned = re.sub(r"</?think\s*/?>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def release_server_cache(self):
        """Ask koboldcpp to abort / drop its KV cache between requests.

        No-op for vLLM (which manages its own KV cache).
        """
        if self.backend != "vllm":
            import requests
            try:
                requests.post(f"{self._base_url}/api/extra/abort", timeout=5)
            except Exception:
                pass

    @staticmethod
    def _is_video(path: str) -> bool:
        return Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def caption_media(self, media_path: str, instruction: str, system_prompt: str | None = None) -> str:
        """Caption a single image or video via one API call.

        For vLLM: passes local file paths directly (requires --allowed-local-media-path /).
        For koboldcpp: base64-encodes downscaled images (no video support).
        """
        import requests

        instruction = self.sanitize_instruction(instruction)
        is_video = self._is_video(media_path)

        user_content = []

        if self.backend == "vllm":
            abs_path = str(Path(media_path).resolve())
            if is_video:
                user_content.append({
                    "type": "video_url",
                    "video_url": {"url": f"file://{abs_path}"},
                })
            else:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{abs_path}"},
                })
        else:
            if is_video:
                raise ValueError("Video captioning requires vLLM backend")
            img_base64, mime_type = self._prepare_image_base64(media_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"},
            })

        user_content.append({"type": "text", "text": instruction})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.get_model_name(),
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if self.backend == "vllm":
            payload["top_k"] = self.top_k
            payload["presence_penalty"] = self.presence_penalty
            payload["repetition_penalty"] = self.repetition_penalty
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            if is_video:
                payload["mm_processor_kwargs"] = {
                    "fps": self.video_fps,
#                    "do_sample_frames": True,
                }
        else:
            payload["cache_prompt"] = False
            payload["stop"] = ["<|im_end|>", "<|endoftext|>"]

        if self.backend != "vllm":
            del img_base64

        session = requests.Session()
        try:
            response = session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=1500,
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()
        finally:
            session.close()
            del payload
            gc.collect()

        return result

    def caption_image(self, image_path: str, instruction: str) -> str:
        """Backward-compatible alias for caption_media."""
        return self.caption_media(image_path, instruction)


# ============================================================================
# Gradio UI
# ============================================================================

captioner: Captioner | None = None


def connect_to_server(
    url: str, port: int, max_tokens: int, temperature: float, top_p: float,
    top_k: int, presence_penalty: float, repetition_penalty: float, max_image_size: int,
    enable_thinking: bool, backend: str, video_fps: float,
) -> str:
    """Connect to server and return status."""
    global captioner
    try:
        captioner = Captioner(
            server_url=url,
            port=int(port),
            max_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_image_size=int(max_image_size),
            enable_thinking=enable_thinking,
            backend=backend,
            video_fps=video_fps,
        )
        model_name = captioner.get_model_name()
        mode = "thinking" if enable_thinking else "instruct"
        return f"Connected — Model: **{model_name}** ({backend}, {mode} mode)"
    except Exception as e:
        captioner = None
        return f"Connection failed: {e}"


def get_media_type(file_path: str) -> str:
    if file_path is None:
        return "none"
    suffix = Path(file_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return "image"
    if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    return "unknown"


def update_preview(file_path: str):
    """Update preview based on uploaded file."""
    if file_path is None:
        return gr.update(visible=False), gr.update(visible=False), "No file selected"

    media_type = get_media_type(file_path)
    path = Path(file_path)

    if media_type == "image":
        from PIL import Image
        with Image.open(path) as img:
            info = f"**{path.name}** — {img.width}x{img.height} {img.format}"
        return gr.update(value=file_path, visible=True), gr.update(visible=False), info

    if media_type == "video":
        size_mb = path.stat().st_size / (1024 * 1024)
        info = f"**{path.name}** — Video ({size_mb:.1f} MB)"
        return gr.update(visible=False), gr.update(value=file_path, visible=True), info

    return gr.update(visible=False), gr.update(visible=False), "Unsupported file type"


def check_instruction_for_thinking_tags(instruction: str):
    """Warn the user if their instruction contains literal thinking tags."""
    if re.search(r"</?think", instruction, re.IGNORECASE):
        return gr.update(
            value=(
                "**Warning:** Your instruction contains `<think>` / `</think>` tags. "
                "These get tokenized as **special tokens** by Qwen3, which causes "
                "koboldcpp to abort generation (EOS triggered on the thinking token). "
                "They will be automatically stripped before sending. "
                "Qwen3 thinking models reason on their own — no prompt needed."
            ),
            visible=True,
        )
    return gr.update(value="", visible=False)


# ---- Single file captioning ------------------------------------------------


def caption_single_file(
    file_path: str,
    instruction: str,
    strip_thinking: bool,
) -> tuple[str, str, str]:
    if captioner is None:
        return "Not connected to server", "", ""
    if file_path is None:
        return "No file selected", "", ""

    media_type = get_media_type(file_path)
    if media_type not in ("image", "video"):
        return "Unsupported file type", "", ""

    if not instruction.strip():
        instruction = Captioner.VIDEO_INSTRUCTION if media_type == "video" else Captioner.IMAGE_INSTRUCTION

    try:
        raw_caption = captioner.caption_media(file_path, instruction)

        json_caption = extract_final_caption(raw_caption) if strip_thinking else raw_caption
        markdown_output = format_caption_as_markdown(Path(file_path).name, raw_caption)
        json_output = json.dumps(
            {"file": Path(file_path).name, "caption": json_caption},
            indent=2,
            ensure_ascii=False,
        )
        return "Caption generated", markdown_output, json_output

    except Exception as e:
        return f"Error: {e}", "", ""


# ---- Looped batch captioning -----------------------------------------------

# Shared flag so the UI "Stop" button can halt a running batch.
_batch_stop = False


def stop_batch():
    global _batch_stop
    _batch_stop = True
    return "Stopping after current image finishes..."


def _load_existing_results(json_path: Path) -> dict[str, dict]:
    """Load already-captioned entries so we can skip / resume."""
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {entry["file"]: entry for entry in data}
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_results(json_path: Path, results: dict[str, dict]):
    """Atomically overwrite the JSON output."""
    items = list(results.values())
    tmp = json_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    tmp.rename(json_path)


def caption_batch(
    files: list,
    instruction: str,
    strip_thinking: bool,
    output_dir: str,
    delay_seconds: float,
    skip_existing: bool,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """Caption images one at a time, saving after every image."""
    global _batch_stop
    _batch_stop = False

    if captioner is None:
        return "Not connected to server", ""
    if not files:
        return "No files selected", ""

    instruction = instruction.strip() or Captioner.DEFAULT_INSTRUCTION

    output_path = Path(output_dir) if output_dir else None
    json_path = output_path / "captions.json" if output_path else None

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if json_path and skip_existing:
        existing = _load_existing_results(json_path)

    results = dict(existing)
    errors: list[str] = []
    skipped = 0
    completed = 0

    for i, file_obj in enumerate(progress.tqdm(files, desc="Captioning")):
        if _batch_stop:
            break

        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        file_name = Path(file_path).name

        if skip_existing and file_name in existing:
            skipped += 1
            continue

        media_type = get_media_type(file_path)
        if media_type not in ("image", "video"):
            errors.append(f"{file_name}: unsupported media type, skipped")
            continue

        try:
            batch_instruction = instruction
            if not batch_instruction.strip():
                batch_instruction = Captioner.VIDEO_INSTRUCTION if media_type == "video" else Captioner.IMAGE_INSTRUCTION
            raw_caption = captioner.caption_media(file_path, batch_instruction)

            final_caption = extract_final_caption(raw_caption) if strip_thinking else raw_caption
            results[file_name] = {"file": file_name, "caption": final_caption}
            completed += 1

            if json_path:
                _save_results(json_path, results)

        except Exception as e:
            errors.append(f"{file_name}: {e}")

        # Force koboldcpp to drop its KV cache and free Python memory
        captioner.release_server_cache()
        gc.collect()

        if delay_seconds > 0 and i < len(files) - 1 and not _batch_stop:
            time.sleep(delay_seconds)

    # Final status
    total = len(files)
    parts = [f"Done — {completed} captioned"]
    if skipped:
        parts.append(f"{skipped} skipped (already done)")
    if errors:
        parts.append(f"{len(errors)} errors")
    parts.append(f"out of {total} files")

    status = ", ".join(parts)
    if json_path:
        status += f"\n\nSaved to `{json_path}`"
    if _batch_stop:
        status += "\n\n**Stopped early by user.**"

    if errors:
        status += "\n\nErrors:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            status += f"\n... and {len(errors) - 10} more"

    # JSON preview — show last 3 results
    preview_items = list(results.values())[-3:]
    json_preview = json.dumps(preview_items, indent=2, ensure_ascii=False)
    n = len(results)
    if n > 3:
        json_preview = f"// showing last 3 of {n} entries\n" + json_preview

    return status, json_preview


# ---- Two-phase batch --------------------------------------------------------


def caption_two_phase_batch(
    files: list,
    config_path: str,
    lexicon_requirements: str,
    custom_instructions: str,
    example_start: str,
    strip_thinking: bool,
    output_dir: str,
    delay_seconds: float,
    skip_existing: bool,
    include_tag_freq: bool,
    tag_freq_top_n: int,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """Two-phase captioning: Agent 1 initial pass, Agent 2 tabula-rasa + reconcile.

    When ``include_tag_freq`` is True, agent_002 receives a running vocabulary
    frequency list built from all agent_001 captions produced so far.  This
    encourages consistent terminology across the batch — important for LoRA
    training where tag entropy hurts effectiveness.
    """
    global _batch_stop
    _batch_stop = False

    if captioner is None:
        return "Not connected to server", ""
    if not files:
        return "No files selected", ""
    if _PromptLoader is None:
        return "prompt_loader not available (pip install pyyaml)", ""

    config_path = (config_path or "").strip()
    if not config_path:
        return "No config.yaml path provided", ""

    try:
        loader = _PromptLoader(config_path)
    except FileNotFoundError as e:
        return str(e), ""
    except Exception as e:
        return f"Failed to load config: {e}", ""

    lexicon = lexicon_requirements.strip()
    custom = custom_instructions.strip()
    example = example_start.strip()
    top_n = int(tag_freq_top_n)

    output_path = Path(output_dir) if output_dir else None
    json_path = output_path / "captions_2phase.json" if output_path else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if json_path and skip_existing:
        existing = _load_existing_results(json_path)

    results = dict(existing)
    # Accumulate all phase-1 captions for running tag frequency
    phase1_captions: list[dict] = []
    # Seed with any existing results so resumed runs start with context
    if include_tag_freq and existing:
        phase1_captions.extend(existing.values())
    errors: list[str] = []
    skipped = 0
    completed = 0

    for i, file_obj in enumerate(progress.tqdm(files, desc="Two-phase captioning")):
        if _batch_stop:
            break

        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        file_name = Path(file_path).name

        if skip_existing and file_name in existing:
            skipped += 1
            continue

        media_type = get_media_type(file_path)
        if media_type not in ("image", "video"):
            errors.append(f"{file_name}: unsupported media type, skipped")
            continue

        try:
            # Phase 1 — initial caption
            system1, user1 = loader.agent1_split(
                lexicon_requirements=lexicon,
                custom_instructions=custom,
                example_start=example,
            )
            raw1 = captioner.caption_media(file_path, user1, system_prompt=system1)
            caption_pass_1 = extract_final_caption(raw1) if strip_thinking else raw1

            # Accumulate for running tag frequency
            phase1_captions.append({"caption": caption_pass_1})

            captioner.release_server_cache()
            gc.collect()

            if _batch_stop:
                errors.append(f"{file_name}: stopped before phase 2")
                break

            # Phase 2 — tabula rasa + critique + reconcile
            system2, user2 = loader.agent2_split(
                caption_pass_1=caption_pass_1,
                lexicon_requirements=lexicon,
                custom_instructions=custom,
                example_start=example,
            )

            # Inject running tag frequency from all phase-1 captions so far
            if include_tag_freq and phase1_captions:
                tag_list = build_tag_frequency(phase1_captions, top_n)
                if tag_list:
                    user2 += (
                        "\n\nCommon vocabulary from this dataset so far — "
                        "use these terms consistently where they apply "
                        "(prefer these over synonyms):\n" + tag_list
                    )

            raw2 = captioner.caption_media(file_path, user2, system_prompt=system2)
            caption_final = extract_final_caption(raw2) if strip_thinking else raw2

            results[file_name] = {"file": file_name, "caption": caption_final}
            completed += 1

            if json_path:
                _save_results(json_path, results)

        except Exception as e:
            errors.append(f"{file_name}: {e}")

        captioner.release_server_cache()
        gc.collect()

        if delay_seconds > 0 and i < len(files) - 1 and not _batch_stop:
            time.sleep(delay_seconds)

    total = len(files)
    parts = [f"Done — {completed} captioned (2-phase)"]
    if skipped:
        parts.append(f"{skipped} skipped (already done)")
    if errors:
        parts.append(f"{len(errors)} errors")
    parts.append(f"out of {total} files")

    status = ", ".join(parts)
    if json_path:
        status += f"\n\nSaved to `{json_path}`"
    if _batch_stop:
        status += "\n\n**Stopped early by user.**"
    if errors:
        status += "\n\nErrors:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            status += f"\n... and {len(errors) - 10} more"

    preview_items = list(results.values())[-3:]
    json_preview = json.dumps(preview_items, indent=2, ensure_ascii=False)
    n = len(results)
    if n > 3:
        json_preview = f"// showing last 3 of {n} entries\n" + json_preview

    return status, json_preview


# ---- Refinement pass --------------------------------------------------------

DEFAULT_REFINEMENT_INSTRUCTION = """\
You are reviewing an image alongside a proposed caption that was written for it.
Your task is to compare the caption against the actual image and produce a
corrected version.

Rules:
- Fix any factual inaccuracies or hallucinated details that do not match the image.
- Add important visible details that the caption missed.
- Remove any descriptions that contradict what is shown.
- Preserve correct details as-is — do not rephrase things that are already accurate.
- Keep the same section format as the original caption.

Proposed caption:
{caption}

Output ONLY the corrected caption (same format, no commentary)."""


def refine_batch(
    json_path: str,
    image_dir: str,
    refinement_instruction: str,
    strip_thinking: bool,
    delay_seconds: float,
    skip_existing: bool,
    include_tag_freq: bool,
    tag_freq_top_n: int,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """Re-caption images using the first-pass caption as context."""
    global _batch_stop
    _batch_stop = False

    if captioner is None:
        return "Not connected to server", ""

    json_path = (json_path or "").strip()
    image_dir = (image_dir or "").strip()
    if not json_path:
        return "No captions JSON path provided", ""
    if not image_dir:
        return "No image directory provided", ""

    src = Path(json_path)
    if not src.exists():
        return f"File not found: {src}", ""

    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        return f"Image directory not found: {img_dir}", ""

    try:
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return f"Failed to read JSON: {e}", ""

    if not isinstance(data, list):
        return "Expected a JSON array at top level", ""

    instruction_template = (
        refinement_instruction.strip() if refinement_instruction and refinement_instruction.strip()
        else DEFAULT_REFINEMENT_INSTRUCTION
    )
    if "{caption}" not in instruction_template:
        return "Refinement instruction must contain `{caption}` placeholder", ""

    # Optionally append a dataset-wide vocabulary list for tag consistency
    if include_tag_freq and data:
        tag_list = build_tag_frequency(data, int(tag_freq_top_n))
        if tag_list:
            instruction_template += (
                "\n\nCommon vocabulary used across this dataset — "
                "use these terms consistently where they apply:\n" + tag_list
            )

    # Output goes next to the source JSON as *_refined.json
    refined_path = src.parent / (src.stem + "_refined.json")

    existing: dict[str, dict] = {}
    if skip_existing:
        existing = _load_existing_results(refined_path)

    results = dict(existing)
    errors: list[str] = []
    skipped = 0
    completed = 0

    for i, entry in enumerate(progress.tqdm(data, desc="Refining")):
        if _batch_stop:
            break

        file_name = entry.get("file", "")
        old_caption = entry.get("caption", "")

        if not file_name or not old_caption:
            errors.append(f"Entry {i}: missing file or caption field")
            continue

        if skip_existing and file_name in existing:
            skipped += 1
            continue

        img_path = img_dir / file_name
        if not img_path.exists():
            errors.append(f"{file_name}: not found in {img_dir}")
            continue

        instruction = instruction_template.replace("{caption}", old_caption)

        try:
            raw = captioner.caption_image(str(img_path), instruction)
            new_caption = extract_final_caption(raw) if strip_thinking else raw
            results[file_name] = {"file": file_name, "caption": new_caption}
            completed += 1

            _save_results(refined_path, results)

        except Exception as e:
            errors.append(f"{file_name}: {e}")

        captioner.release_server_cache()
        gc.collect()

        if delay_seconds > 0 and i < len(data) - 1 and not _batch_stop:
            time.sleep(delay_seconds)

    # Status
    total = len(data)
    parts = [f"Done — {completed} refined"]
    if skipped:
        parts.append(f"{skipped} skipped (already refined)")
    if errors:
        parts.append(f"{len(errors)} errors")
    parts.append(f"out of {total} entries")

    status = ", ".join(parts)
    status += f"\n\nSaved to `{refined_path}`"
    if _batch_stop:
        status += "\n\n**Stopped early by user.**"
    if errors:
        status += "\n\nErrors:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            status += f"\n... and {len(errors) - 10} more"

    preview_items = list(results.values())[-3:]
    json_preview = json.dumps(preview_items, indent=2, ensure_ascii=False)
    n = len(results)
    if n > 3:
        json_preview = f"// showing last 3 of {n} entries\n" + json_preview

    return status, json_preview


# ---- Tag frequency analysis -------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "it",
    "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "them", "their", "there", "here", "what", "which",
    "who", "whom", "when", "where", "why", "how", "all", "both", "each",
    "none", "not", "no", "nor", "so", "yet", "while", "if", "then",
    "than", "also", "just", "very", "more", "one", "two", "three",
    "s", "t", "ve", "re", "ll", "d", "m",
}


def build_tag_frequency(data: list[dict], top_n: int = 100) -> str:
    """Return a frequency-sorted vocabulary list extracted from captions.

    Strips section headers (e.g. ``[VISUAL]:``) and common stop words,
    then returns the top-N remaining terms as a comma-separated string
    suitable for injection into a refinement prompt.
    """
    from collections import Counter

    counter: Counter = Counter()
    for entry in data:
        caption = entry.get("caption", "")
        # Remove structured section headers like [VISUAL]:
        text = re.sub(r"\[[A-Z]+\]:\s*", " ", caption)
        # Tokenize: lowercase words of 3+ chars only
        words = re.findall(r"[a-z]{3,}", text.lower())
        counter.update(w for w in words if w not in _STOP_WORDS)

    top = [word for word, _ in counter.most_common(top_n)]
    return ", ".join(top)


# ---- Export captions.json -> individual .txt files --------------------------


def export_json_to_txt(json_path: str, txt_output_dir: str) -> str:
    """Read a captions.json and write one .txt per image."""
    json_path = (json_path or "").strip()
    if not json_path:
        return "No JSON path provided"

    src = Path(json_path)
    if not src.exists():
        return f"File not found: {src}"

    try:
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return f"Failed to read JSON: {e}"

    if not isinstance(data, list):
        return "Expected a JSON array at top level"

    dest = Path(txt_output_dir.strip()) if txt_output_dir and txt_output_dir.strip() else src.parent
    dest.mkdir(parents=True, exist_ok=True)

    written = 0
    errors: list[str] = []

    for entry in data:
        try:
            image_name = entry["file"]
            caption = entry["caption"]
        except (KeyError, TypeError) as e:
            errors.append(f"Malformed entry: {e}")
            continue

        txt_path = dest / (Path(image_name).stem + ".txt")
        try:
            txt_path.write_text(caption, encoding="utf-8")
            written += 1
        except OSError as e:
            errors.append(f"{txt_path.name}: {e}")

    status = f"Wrote {written} .txt file(s) to `{dest}`"
    if errors:
        status += "\n\nErrors:\n" + "\n".join(errors[:10])
    return status


# ============================================================================
# Build the interface
# ============================================================================


def load_twophase_config(config_path: str) -> tuple[str, str, str, str]:
    """Load user settings from a config.yaml into the Two-Phase UI fields.

    Returns (status, lexicon_requirements, custom_instructions, example_start).
    """
    config_path = (config_path or "").strip()
    if not config_path:
        return "No config path provided", "", "", ""

    path = Path(config_path)
    if not path.exists():
        return f"File not found: {path}", "", "", ""

    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        return "PyYAML not installed (pip install pyyaml)", "", "", ""
    except Exception as e:
        return f"Failed to load: {e}", "", "", ""

    if not isinstance(cfg, dict):
        return "Invalid YAML (expected a mapping at top level)", "", "", ""

    lexicon = str(cfg.get("lexicon_requirements", "")).strip()
    custom = str(cfg.get("custom_instructions", "")).strip()
    example = str(cfg.get("example_start", "")).strip()

    loaded = []
    if lexicon:
        loaded.append("lexicon_requirements")
    if custom:
        loaded.append("custom_instructions")
    if example:
        loaded.append("example_start")

    if loaded:
        status = f"Loaded: {', '.join(loaded)}"
    else:
        status = "Config loaded but no user settings found (lexicon_requirements, custom_instructions, example_start)"

    return status, lexicon, custom, example


_UI_CSS = """
.container { max-width: 1200px; margin: auto; }
.preview-box {
    border: 2px solid #3b3b3b;
    border-radius: 8px;
    padding: 10px;
    background: #1a1a1a;
    min-height: 300px;
}
.output-box {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
}
"""

_UI_THEME = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0d0d0d",
    body_background_fill_dark="#0d0d0d",
    block_background_fill="#1a1a1a",
    block_background_fill_dark="#1a1a1a",
    border_color_primary="#2d2d2d",
    border_color_primary_dark="#2d2d2d",
)


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Lora Caption Tool") as demo:
        gr.Markdown("# Lora Media Captioner\nOne-at-a-time captioning for large VL models via vLLM / koboldcpp / OpenAI-compatible servers.")

        with gr.Row():
            # ── Left column: connection & settings ──────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Server Connection")

                with gr.Group():
                    backend = gr.Radio(
                        label="Backend",
                        choices=["vllm", "koboldcpp"],
                        value="vllm",
                        info="vLLM: Qwen3.5 with video+image support. koboldcpp: image-only legacy mode.",
                    )
                    server_url = gr.Textbox(
                        label="Server URL",
                        value="http://localhost",
                    )
                    server_port = gr.Number(label="Port", value=8000, precision=0)
                    max_tokens = gr.Number(
                        label="Max Tokens",
                        value=4096,
                        precision=0,
                        info="Qwen3.5 thinking mode benefits from higher token budgets (32768 for complex tasks)",
                    )
                    connect_btn = gr.Button("Connect", variant="primary")
                    connection_status = gr.Markdown("Not connected")

                gr.Markdown("### Sampling Parameters")
                with gr.Group():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        info="Qwen3.5 thinking: 1.0, instruct: 0.7",
                    )
                    top_p = gr.Slider(
                        label="Top-P",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                    )
                    top_k = gr.Number(
                        label="Top-K",
                        value=20,
                        precision=0,
                    )
                    presence_penalty = gr.Slider(
                        label="Presence Penalty",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.5,
                        step=0.1,
                        info="Reduces repetition. Qwen3.5 recommends 1.5",
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        info="1.0 = off. Reduces repetition. Try 1.1–1.3 for Qwen3.",
                    )

                gr.Markdown("### Caption Settings")

                with gr.Group():
                    enable_thinking = gr.Checkbox(
                        label="Enable thinking mode",
                        value=True,
                        info="Qwen3.5 thinks before responding. Produces better captions but uses more tokens.",
                    )
                    strip_thinking = gr.Checkbox(
                        label="Strip thinking/reasoning from output",
                        value=True,
                        info="Remove <think>...</think> from saved captions",
                    )
                    max_image_size = gr.Slider(
                        label="Max Image Size (px, koboldcpp only)",
                        minimum=256,
                        maximum=2048,
                        value=768,
                        step=64,
                        info="Downscale before sending (koboldcpp only, vLLM uses file paths)",
                    )
                    video_fps = gr.Slider(
                        label="Video FPS (vLLM only)",
                        minimum=0.5,
                        maximum=10.0,
                        value=2.0,
                        step=0.5,
                        info="Frame sampling rate for video captioning",
                    )
                    instruction = gr.Textbox(
                        label="Custom Instruction",
                        placeholder="Leave empty for default (auto-selects image vs video instruction)...",
                        lines=4,
                    )
                    instruction_warning = gr.Markdown("", visible=False)

            # ── Right column: preview ───────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Media Preview")

                with gr.Group(elem_classes="preview-box"):
                    media_info = gr.Markdown("No file selected")
                    image_preview = gr.Image(
                        label="Preview",
                        visible=False,
                        height=350,
                    )
                    video_preview = gr.Video(
                        label="Video Preview",
                        visible=False,
                        height=350,
                    )

        gr.Markdown("---")

        with gr.Tabs():
            # ── Single file tab ─────────────────────────────────────────
            with gr.Tab("Single File"):
                with gr.Row():
                    with gr.Column():
                        single_file = gr.File(
                            label="Upload Image or Video",
                            file_types=["image", "video"],
                        )
                        caption_btn = gr.Button(
                            "Generate Caption",
                            variant="primary",
                            size="lg",
                        )
                        single_status = gr.Markdown("")

                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab("Formatted"):
                                markdown_output = gr.Markdown(
                                    label="Formatted Caption",
                                    elem_classes="output-box",
                                )
                            with gr.Tab("JSON"):
                                json_output = gr.Code(
                                    label="JSON Output",
                                    language="json",
                                    elem_classes="output-box",
                                )

            # ── Batch (looped) tab ──────────────────────────────────────
            with gr.Tab("Batch (Looped)"):
                gr.Markdown(
                    "Upload multiple images — they are sent to the server **one at a time** "
                    "and each caption is saved to disk immediately.  If the server crashes "
                    "mid-batch, re-run with **Skip already captioned** checked to resume."
                )
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(
                            label="Upload Images & Videos",
                            file_count="multiple",
                            file_types=["image", "video"],
                        )
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            placeholder="/path/to/output/",
                            info="captions.json is written here (created if needed)",
                        )
                        with gr.Row():
                            delay_seconds = gr.Number(
                                label="Delay between images (sec)",
                                value=1.0,
                                precision=1,
                                info="Gives the backend time to free memory",
                            )
                            skip_existing = gr.Checkbox(
                                label="Skip already captioned",
                                value=True,
                                info="Resume from where you left off",
                            )
                        with gr.Row():
                            batch_btn = gr.Button(
                                "Start Batch",
                                variant="primary",
                                size="lg",
                            )
                            stop_btn = gr.Button(
                                "Stop",
                                variant="stop",
                                size="lg",
                            )

                    with gr.Column():
                        batch_status = gr.Markdown("")
                        batch_preview = gr.Code(
                            label="JSON Preview",
                            language="json",
                            elem_classes="output-box",
                        )

            # ── Two-Phase Batch tab ─────────────────────────────────────────
            with gr.Tab("Two-Phase Batch"):
                gr.Markdown(
                    "**Agent 1** produces an initial caption (Phase 1). "
                    "**Agent 2** writes a fresh independent caption, critiques Phase 1, "
                    "then reconciles both into `{CAPTION_FINAL}`.  \n"
                    "Requires a `config.yaml` — copy from `config.yaml.template` and fill in your values."
                )
                with gr.Row():
                    with gr.Column():
                        twophase_files = gr.File(
                            label="Upload Images & Videos",
                            file_count="multiple",
                            file_types=["image", "video"],
                        )
                        with gr.Row():
                            twophase_config = gr.Textbox(
                                label="config.yaml Path",
                                placeholder="/path/to/Lora/scripts/config.yaml",
                                info="Copy config.yaml.template → config.yaml and fill in your values",
                                scale=3,
                            )
                            twophase_load_btn = gr.Button(
                                "Load",
                                variant="secondary",
                                size="sm",
                                scale=1,
                            )
                        twophase_load_status = gr.Markdown("")
                        twophase_lexicon = gr.Textbox(
                            label="Lexicon Requirements",
                            placeholder="Vocabulary and style rules for your dataset...",
                            lines=4,
                        )
                        twophase_custom = gr.Textbox(
                            label="Custom Instructions",
                            placeholder="Per-run overrides (leave empty if none)...",
                            lines=2,
                        )
                        twophase_example = gr.Textbox(
                            label="Example Start",
                            placeholder="woman, ",
                            info="Short example of how your tags should open",
                        )
                        twophase_output_dir = gr.Textbox(
                            label="Output Directory",
                            placeholder="/path/to/output/",
                            info="captions_2phase.json is written here",
                        )
                        with gr.Row():
                            twophase_delay = gr.Number(
                                label="Delay between files (sec)",
                                value=1.0,
                                precision=1,
                            )
                            twophase_skip = gr.Checkbox(
                                label="Skip already captioned",
                                value=True,
                            )
                        twophase_strip = gr.Checkbox(
                            label="Strip thinking/reasoning",
                            value=True,
                        )
                        with gr.Row():
                            twophase_tag_freq = gr.Checkbox(
                                label="Running tag frequency for Agent 2",
                                value=True,
                                info="Injects accumulated Phase 1 vocabulary into Agent 2 to consolidate tags",
                            )
                            twophase_tag_top_n = gr.Number(
                                label="Top N tags",
                                value=100,
                                precision=0,
                                info="How many of the most frequent terms to inject",
                            )
                        with gr.Row():
                            twophase_btn = gr.Button(
                                "Start Two-Phase Batch",
                                variant="primary",
                                size="lg",
                            )
                            twophase_stop_btn = gr.Button(
                                "Stop",
                                variant="stop",
                                size="lg",
                            )

                    with gr.Column():
                        twophase_status = gr.Markdown("")
                        twophase_preview = gr.Code(
                            label="JSON Preview",
                            language="json",
                            elem_classes="output-box",
                        )

            # ── Refinement tab ─────────────────────────────────────────────
            with gr.Tab("Refinement"):
                gr.Markdown(
                    "Re-examine each image alongside its first-pass caption.  "
                    "The model sees the image and the caption as a fresh review "
                    "task — it is **not** told it authored the original.  "
                    "Output is saved as `<name>_refined.json`."
                )
                with gr.Row():
                    with gr.Column():
                        refine_json_path = gr.Textbox(
                            label="Source captions.json",
                            placeholder="/path/to/captions.json",
                            info="The first-pass captions to refine",
                        )
                        refine_image_dir = gr.Textbox(
                            label="Image Directory",
                            placeholder="/path/to/images/",
                            info="Folder containing the original images",
                        )
                        refine_instruction = gr.Textbox(
                            label="Refinement Instruction",
                            value=DEFAULT_REFINEMENT_INSTRUCTION,
                            lines=10,
                            info="Must contain {caption} placeholder",
                        )
                        refine_strip_thinking = gr.Checkbox(
                            label="Strip thinking/reasoning",
                            value=True,
                        )
                        with gr.Row():
                            include_tag_freq = gr.Checkbox(
                                label="Include tag frequency context",
                                value=True,
                                info="Injects the most common vocabulary from the dataset into the prompt to enforce consistent terminology",
                            )
                            tag_freq_top_n = gr.Number(
                                label="Top N tags",
                                value=100,
                                precision=0,
                                info="How many of the most frequent terms to include",
                            )
                        with gr.Row():
                            refine_delay = gr.Number(
                                label="Delay between images (sec)",
                                value=1.0,
                                precision=1,
                            )
                            refine_skip_existing = gr.Checkbox(
                                label="Skip already refined",
                                value=True,
                                info="Resume from where you left off",
                            )
                        with gr.Row():
                            refine_btn = gr.Button(
                                "Start Refinement",
                                variant="primary",
                                size="lg",
                            )
                            refine_stop_btn = gr.Button(
                                "Stop",
                                variant="stop",
                                size="lg",
                            )

                    with gr.Column():
                        refine_status = gr.Markdown("")
                        refine_preview = gr.Code(
                            label="Refined JSON Preview",
                            language="json",
                            elem_classes="output-box",
                        )

            # ── Export tab ──────────────────────────────────────────────────
            with gr.Tab("Export to .txt"):
                gr.Markdown(
                    "Convert a `captions.json` into individual `.txt` files "
                    "named after each image (e.g. `photo_01.jpg` → `photo_01.txt`).  "
                    "Most LoRA trainers expect captions in this format."
                )
                with gr.Row():
                    with gr.Column():
                        export_json_path = gr.Textbox(
                            label="captions.json Path",
                            placeholder="/path/to/captions.json",
                        )
                        export_output_dir = gr.Textbox(
                            label="Output Directory (optional)",
                            placeholder="Leave empty to write next to the JSON",
                        )
                        export_btn = gr.Button(
                            "Export .txt Files",
                            variant="primary",
                            size="lg",
                        )
                    with gr.Column():
                        export_status = gr.Markdown("")

        # ── Event wiring ────────────────────────────────────────────────
        connect_btn.click(
            fn=connect_to_server,
            inputs=[
                server_url, server_port, max_tokens, temperature, top_p,
                top_k, presence_penalty, repetition_penalty, max_image_size, enable_thinking,
                backend, video_fps,
            ],
            outputs=[connection_status],
        )

        instruction.change(
            fn=check_instruction_for_thinking_tags,
            inputs=[instruction],
            outputs=[instruction_warning],
        )

        single_file.change(
            fn=update_preview,
            inputs=[single_file],
            outputs=[image_preview, video_preview, media_info],
        )

        caption_btn.click(
            fn=caption_single_file,
            inputs=[single_file, instruction, strip_thinking],
            outputs=[single_status, markdown_output, json_output],
        )

        batch_btn.click(
            fn=caption_batch,
            inputs=[
                batch_files,
                instruction,
                strip_thinking,
                output_dir,
                delay_seconds,
                skip_existing,
            ],
            outputs=[batch_status, batch_preview],
        )

        stop_btn.click(fn=stop_batch, outputs=[batch_status])

        refine_btn.click(
            fn=refine_batch,
            inputs=[
                refine_json_path,
                refine_image_dir,
                refine_instruction,
                refine_strip_thinking,
                refine_delay,
                refine_skip_existing,
                include_tag_freq,
                tag_freq_top_n,
            ],
            outputs=[refine_status, refine_preview],
        )

        refine_stop_btn.click(fn=stop_batch, outputs=[refine_status])

        twophase_load_btn.click(
            fn=load_twophase_config,
            inputs=[twophase_config],
            outputs=[twophase_load_status, twophase_lexicon, twophase_custom, twophase_example],
        )

        twophase_btn.click(
            fn=caption_two_phase_batch,
            inputs=[
                twophase_files,
                twophase_config,
                twophase_lexicon,
                twophase_custom,
                twophase_example,
                twophase_strip,
                twophase_output_dir,
                twophase_delay,
                twophase_skip,
                twophase_tag_freq,
                twophase_tag_top_n,
            ],
            outputs=[twophase_status, twophase_preview],
        )

        twophase_stop_btn.click(fn=stop_batch, outputs=[twophase_status])

        export_btn.click(
            fn=export_json_to_txt,
            inputs=[export_json_path, export_output_dir],
            outputs=[export_status],
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lora Image Captioner")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-port", type=int, default=8000, help="vLLM/backend port")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=_UI_THEME,
        css=_UI_CSS,
    )


if __name__ == "__main__":
    main()
