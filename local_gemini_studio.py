# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2,<2", "google-genai>=2.25.0", "pillow"]
# ///
"""
Gemini Studio MCP: lets Claude make images, videos, speech and music with
Google's latest models (Nano Banana 2.1, Gemini Omni Flash, Gemini 3.8 TTS, Lyria 3.5)
through your Google AI Studio API key.

Everything is saved to OUTPUT_DIR. Every result also returns an interaction_id;
pass it back as `continue_from` to edit the same image/video in follow-up turns
("make it night", "move the character left") with consistent results.

Env vars:
  GEMINI_API_KEY   required, from aistudio.google.com
  OUTPUT_DIR       default ~/gemini-studio-output
  IMAGE_MODEL      default gemini-nano-banana-2.1
  VIDEO_MODEL      default gemini-omni-1.1-flash
  TTS_MODEL        default gemini-3.8-flash-tts
  MUSIC_MODEL      default lyria-3.5
"""

from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
import os
import re
import time
from datetime import datetime
from pathlib import Path

from google import genai
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-nano-banana-2.1")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "gemini-omni-1.1-flash")
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-3.8-flash-tts")
MUSIC_MODEL = os.environ.get("MUSIC_MODEL", "lyria-3.5")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path.home() / "gemini-studio-output")).expanduser()

mcp = FastMCP("gemini-studio")
_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to the MCP server's env in your Claude config.")
        _client = genai.Client(api_key=key)
    return _client


# ---------- helpers ----------

def _out_path(name: str | None, prefix: str, ext: str, subdir: str) -> Path:
    folder = OUTPUT_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    if name:
        stem = re.sub(r"[^\w\-]+", "_", Path(name).stem).strip("_") or prefix
    else:
        stem = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    path = folder / f"{stem}.{ext}"
    i = 2
    while path.exists():
        path = folder / f"{stem}_{i}.{ext}"
        i += 1
    return path


def _image_part(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Reference image not found: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return {"type": "image", "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": mime}


def _preview(data: bytes, max_side: int = 640) -> Image:
    """Small JPEG so Claude can see the result without flooding its context."""
    im = PILImage.open(io.BytesIO(data))
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=80)
    return Image(data=buf.getvalue(), format="jpeg")


def _state(f) -> str:
    s = getattr(f, "state", None)
    return str(getattr(s, "name", s) or "").upper()


def _wait_active(name: str, timeout: int = 600) -> None:
    start = time.time()
    while True:
        st = _state(client().files.get(name=name))
        if "ACTIVE" in st:
            return
        if "FAILED" in st:
            raise RuntimeError(f"File {name} failed processing")
        if time.time() - start > timeout:
            raise TimeoutError(f"File {name} not ready after {timeout}s")
        time.sleep(5)


def _download(uri_or_file, dest: Path) -> None:
    res = client().files.download(file=uri_or_file, destination=str(dest))
    if isinstance(res, (bytes, bytearray)) and not dest.exists():
        dest.write_bytes(res)


def _no_output_msg(interaction, kind: str) -> str:
    txt = getattr(interaction, "output_text", None)
    return f"No {kind} returned (possibly blocked by safety filters). Model said: {txt or 'nothing'}"


def _err(e: Exception) -> str:
    return f"Error: {type(e).__name__}: {e}"


# ---------- tools ----------

@mcp.tool()
async def generate_image(
    prompt: str,
    aspect_ratio: str = "9:16",
    image_size: str = "2K",
    reference_images: list[str] | None = None,
    continue_from: str | None = None,
    filename: str | None = None,
    model: str | None = None,
    thinking_level: str | None = None,
) -> list:
    """Generate or edit an image (Nano Banana 2.1 by default).

    prompt: scene description, or the edit to make.
    aspect_ratio: e.g. "9:16" (shorts), "16:9", "1:1", "4:5", "3:4".
    image_size: "1K", "2K" or "4K" (uppercase K).
    reference_images: local file paths (up to 14) - character sheets, style refs,
        or an image to edit. Use the same character refs every scene for consistency.
    continue_from: interaction_id from a previous result, to refine that same image.
    filename: optional name, e.g. "scene_03".
    model: override, e.g. "gemini-3-pro-image" (Nano Banana Pro) or "gemini-3.1-flash-image".
    thinking_level: optional "low" / "high" for harder compositions.
    """
    def run():
        parts = [_image_part(p) for p in (reference_images or [])]
        inp = [{"type": "text", "text": prompt}, *parts] if parts else prompt
        kwargs = dict(
            model=model or IMAGE_MODEL,
            input=inp,
            response_format={"type": "image", "mime_type": "image/png",
                             "aspect_ratio": aspect_ratio, "image_size": image_size.upper()},
        )
        if continue_from:
            kwargs["previous_interaction_id"] = continue_from
        if thinking_level:
            kwargs["generation_config"] = {"thinking_level": thinking_level}
        return client().interactions.create(**kwargs)

    try:
        it = await asyncio.to_thread(run)
        img = getattr(it, "output_image", None)
        if not img or not getattr(img, "data", None):
            return [_no_output_msg(it, "image")]
        data = base64.b64decode(img.data)
        path = _out_path(filename, "image", "png", "images")
        path.write_bytes(data)
        return [f"Saved: {path}\ninteraction_id: {it.id}", _preview(data)]
    except Exception as e:
        return [_err(e)]


@mcp.tool()
async def generate_video(
    prompt: str,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    images: list[str] | None = None,
    video: str | None = None,
    continue_from: str | None = None,
    task: str | None = None,
    filename: str | None = None,
    model: str | None = None,
) -> str:
    """Generate, animate, edit or extend a video with Gemini Omni Flash. Takes a few minutes.

    prompt: what happens in the shot (put negatives here too, e.g. "no text on screen").
    aspect_ratio: "9:16" or "16:9".
    resolution: "360p", "720p", "1080p" or "4k" (1080p/4k are upscaled, slower).
    images: local paths. 1 image = animate it; 2 images = first frame -> last frame;
        several = subject/character references.
    video: local .mp4 (<=10s) to edit or extend ("Continue the scene.").
    continue_from: interaction_id of a previous video to edit/extend it in a new turn.
    task: optional hint: "text_to_video", "image_to_video", "reference_to_video", "edit", "extend".
    filename: optional name, e.g. "scene_03".
    """
    def run():
        c = client()
        parts: list[dict] = [_image_part(p) for p in (images or [])]
        if video:
            vp = Path(video).expanduser()
            if not vp.exists():
                raise FileNotFoundError(f"Video not found: {vp}")
            up = c.files.upload(file=str(vp))
            _wait_active(up.name)
            parts.append({"type": "video", "uri": c.files.get(name=up.name).uri})
        inp = [*parts, {"type": "text", "text": prompt}] if parts else prompt
        kwargs = dict(
            model=model or VIDEO_MODEL,
            input=inp,
            response_format={"type": "video", "delivery": "uri",
                             "aspect_ratio": aspect_ratio, "resolution": resolution},
        )
        if continue_from:
            kwargs["previous_interaction_id"] = continue_from
        if task:
            kwargs["generation_config"] = {"video_config": {"task": task}}
        it = c.interactions.create(**kwargs)
        out = getattr(it, "output_video", None)
        if not out:
            return it, None
        path = _out_path(filename, "video", "mp4", "videos")
        if getattr(out, "uri", None):
            _wait_active(f"files/{out.uri.rstrip('/').split('/')[-1]}")
            _download(out.uri, path)
        elif getattr(out, "data", None):
            path.write_bytes(base64.b64decode(out.data))
        else:
            return it, None
        return it, path

    try:
        it, path = await asyncio.to_thread(run)
        if not path:
            return _no_output_msg(it, "video")
        return f"Saved: {path}\ninteraction_id: {it.id}"
    except Exception as e:
        return _err(e)


@mcp.tool()
async def generate_speech(
    text: str | None = None,
    voice: str = "Kore",
    style: str | None = None,
    turns: list[dict] | None = None,
    speaker_voices: dict[str, str] | None = None,
    filename: str | None = None,
    model: str | None = None,
) -> str:
    """Text-to-speech with Gemini 3.8 Flash TTS. Saves a 24kHz mono WAV.

    Single speaker: pass `text`, `voice`, optional `style` (e.g. "excited, fast, Gen Z").
      Inline tags work inside text: <laugh> <sigh> <gasp> <short pause> <long pause>;
      CAPS = emphasis. Keep text as pure transcript (stage directions get read aloud).
    Two speakers: pass `turns` = [{"speaker": "Dorjee", "text": "...", "style": "..."}, ...]
      and `speaker_voices` = {"Dorjee": "Puck", "Pema": "Kore"} (max 2 speakers).
    Voices: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe,
      Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi,
      Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird,
      Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat.
    """
    def run():
        if turns:
            content = []
            for t in turns:
                ann = {"type": "speech_metadata", "speaker": t["speaker"]}
                if t.get("style"):
                    ann["style"] = t["style"]
                content.append({"type": "text", "text": t["text"], "annotations": [ann]})
            voices = speaker_voices or {}
            names = list(dict.fromkeys(t["speaker"] for t in turns))
            defaults = ["Puck", "Kore"]
            speech_config = {"speakers": [
                {"speaker": n, "voice": voices.get(n, defaults[i % 2])} for i, n in enumerate(names)
            ]}
        else:
            if not text:
                raise ValueError("Pass either `text` or `turns`.")
            item = {"type": "text", "text": text}
            if style:
                item["annotations"] = [{"type": "speech_metadata", "style": style}]
            content = [item]
            speech_config = [{"voice": voice}]
        return client().interactions.create(
            model=model or TTS_MODEL,
            input=[{"type": "user_input", "content": content}],
            response_format={"type": "audio"},
            generation_config={"speech_config": speech_config},
        )

    try:
        it = await asyncio.to_thread(run)
        audio = getattr(it, "output_audio", None)
        if not audio or not getattr(audio, "data", None):
            return _no_output_msg(it, "audio")
        path = _out_path(filename, "speech", "wav", "audio")
        path.write_bytes(base64.b64decode(audio.data))
        return f"Saved: {path}"
    except Exception as e:
        return _err(e)


@mcp.tool()
async def generate_music(prompt: str, filename: str | None = None, model: str | None = None) -> str:
    """Generate music with Lyria 3.5 (MP3).

    No duration/instrumental flags: say it in the prompt, e.g.
    "30-second lo-fi Himalayan flute beat, instrumental only, no vocals".
    """
    def run():
        return client().interactions.create(
            model=model or MUSIC_MODEL, input=prompt, response_format={"type": "audio"},
        )

    try:
        it = await asyncio.to_thread(run)
        audio = getattr(it, "output_audio", None)
        if not audio or not getattr(audio, "data", None):
            return _no_output_msg(it, "music")
        path = _out_path(filename, "music", "mp3", "music")
        path.write_bytes(base64.b64decode(audio.data))
        lyrics = getattr(it, "output_text", None)
        return f"Saved: {path}" + (f"\nLyrics/structure:\n{lyrics}" if lyrics else "")
    except Exception as e:
        return _err(e)


@mcp.tool()
def list_outputs(kind: str = "all", limit: int = 20) -> str:
    """List recently generated files. kind: "images", "videos", "audio", "music" or "all"."""
    dirs = ["images", "videos", "audio", "music"] if kind == "all" else [kind]
    files = [f for d in dirs if (OUTPUT_DIR / d).exists() for f in (OUTPUT_DIR / d).iterdir() if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return f"No outputs yet in {OUTPUT_DIR}"
    return "\n".join(str(f) for f in files[:limit])


if __name__ == "__main__":
    mcp.run()
