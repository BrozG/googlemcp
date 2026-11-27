# gemini-studio MCP

A remote MCP server that lets Claude generate with Google's latest AI Studio models. Runs on Vercel.

| tool | default model | does |
|---|---|---|
| `generate_image` | `gemini-nano-banana-2.1` | text→image, edit, up to 14 reference images, multi-turn refine |
| `generate_video` | `gemini-omni-1.1-flash` | text→video, animate image, first→last frame, edit/extend clips |
| `generate_speech` | `gemini-3.8-flash-tts` | 1–2 speaker TTS, style + `<laugh>`/`<sigh>` tags, WAV |
| `generate_music` | `lyria-3.5` | music/songs, MP3 |

Outputs are uploaded to **Vercel Blob** and returned as public URLs. Image/video results also return an `interaction_id`; pass it back as `continue_from` to edit the same output.

## Deploy

1. Import this repo in Vercel (New Project → pick the repo). Framework: Next.js, defaults are fine.
2. **Storage → Create → Blob** and connect it to the project. This adds `BLOB_READ_WRITE_TOKEN` automatically.
3. **Settings → Environment Variables**:
   - `GEMINI_API_KEY`: from aistudio.google.com
   - `MCP_SECRET`: a long random string (e.g. `openssl rand -hex 24`)
4. Make sure **Fluid compute** is on (default for new projects). Redeploy.

## Connect Claude

**claude.ai / Claude app:** Settings → Connectors → Add custom connector →
```
https://YOUR-PROJECT.vercel.app/api/mcp?key=YOUR_MCP_SECRET
```

**Claude Code:**
```
claude mcp add --transport http gemini-studio "https://YOUR-PROJECT.vercel.app/api/mcp?key=YOUR_MCP_SECRET"
```

Anyone with that URL can spend your Gemini credits. Keep it private, and rotate `MCP_SECRET` if it leaks.

## Limits

- Vercel Hobby functions max out at 300s. Omni video at 720p usually fits. 1080p/4k (upscaled) may time out: use Pro (set `maxDuration = 800` in `app/api/mcp/route.ts`) or the local version.
- Reference images and videos are passed as **URLs** (or `data:` URIs), since the server can't see your disk. Blob URLs from earlier results work directly.
- Uploaded clips for edit/extend must be ≤10s.
- Heavier models need billing enabled on your AI Studio key.

## Local version

`local_gemini_studio.py` is the same toolset as a local stdio MCP (saves files to disk, accepts local paths). Run it with `uv run --script local_gemini_studio.py` and give it `GEMINI_API_KEY` in your Claude Desktop config.
