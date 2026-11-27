import { createMcpHandler } from "mcp-handler";
import { z } from "zod";
import {
  IMAGE_MODEL, VIDEO_MODEL, TTS_MODEL, MUSIC_MODEL,
  interact, store, imagePart, preview, uploadVideoFromUrl, downloadFile, noOutput, errText,
} from "@/lib/gemini";

// Video generation takes minutes. Hobby + Fluid compute allows up to 300s; Pro up to 800s.
export const maxDuration = 300;
export const runtime = "nodejs";

const text = (t: string) => ({ content: [{ type: "text" as const, text: t }] });

const VOICES =
  "Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, " +
  "Algieba, Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, " +
  "Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat";

const handler = createMcpHandler(
  (server) => {
    server.registerTool(
      "generate_image",
      {
        title: "Generate image (Nano Banana)",
        description:
          "Generate or edit an image with Nano Banana 2.1 (default). Returns a public URL, an interaction_id and a preview. " +
          "Pass interaction_id back as continue_from to refine the same image. Use the same reference_images " +
          "(character sheets) every scene to keep characters consistent.",
        inputSchema: z.object({
          prompt: z.string().describe("Scene description, or the edit to make"),
          aspect_ratio: z.string().default("9:16").describe('"9:16", "16:9", "1:1", "4:5", "3:4" ...'),
          image_size: z.enum(["1K", "2K", "4K"]).default("2K"),
          reference_images: z.array(z.string()).optional()
            .describe("Up to 14 image URLs or data: URIs (characters, style refs, image to edit)"),
          continue_from: z.string().optional().describe("interaction_id of a previous image to refine"),
          filename: z.string().optional(),
          model: z.string().optional().describe('e.g. "gemini-3-pro-image" (Nano Banana Pro)'),
          thinking_level: z.enum(["low", "high"]).optional(),
        }),
      },
      async (a) => {
        try {
          const parts = await Promise.all((a.reference_images ?? []).map(imagePart));
          const params: Record<string, unknown> = {
            model: a.model ?? IMAGE_MODEL,
            input: parts.length ? [{ type: "text", text: a.prompt }, ...parts] : a.prompt,
            response_format: { type: "image", mime_type: "image/png", aspect_ratio: a.aspect_ratio, image_size: a.image_size },
          };
          if (a.continue_from) params.previous_interaction_id = a.continue_from;
          if (a.thinking_level) params.generation_config = { thinking_level: a.thinking_level };
          const it = await interact(params);
          if (!it?.output_image?.data) return text(noOutput(it, "image"));
          const buf = Buffer.from(it.output_image.data, "base64");
          const url = await store(buf, "images", a.filename, "image", "png", "image/png");
          return {
            content: [
              { type: "text" as const, text: `URL: ${url}\ninteraction_id: ${it.id}` },
              { type: "image" as const, data: await preview(buf), mimeType: "image/jpeg" },
            ],
          };
        } catch (e) {
          return text(errText(e));
        }
      },
    );

    server.registerTool(
      "generate_video",
      {
        title: "Generate video (Gemini Omni)",
        description:
          "Generate, animate, edit or extend a video with Gemini Omni Flash. Takes minutes. " +
          "1 image = animate it; 2 images = first frame to last frame; several = subject references. " +
          "Pass a video URL (<=10s) to edit/extend it. Put negatives in the prompt.",
        inputSchema: z.object({
          prompt: z.string(),
          aspect_ratio: z.enum(["9:16", "16:9"]).default("9:16"),
          resolution: z.enum(["360p", "720p", "1080p", "4k"]).default("720p")
            .describe("1080p/4k are upscaled and slower; may exceed the time limit on Hobby"),
          images: z.array(z.string()).optional().describe("Image URLs or data: URIs"),
          video: z.string().optional().describe("URL of an .mp4 (<=10s) to edit or extend"),
          continue_from: z.string().optional().describe("interaction_id of a previous video"),
          task: z.enum(["text_to_video", "image_to_video", "reference_to_video", "edit", "extend"]).optional(),
          filename: z.string().optional(),
          model: z.string().optional(),
        }),
      },
      async (a) => {
        try {
          const parts: Record<string, unknown>[] = await Promise.all((a.images ?? []).map(imagePart));
          if (a.video) parts.push({ type: "video", uri: await uploadVideoFromUrl(a.video) });
          const params: Record<string, unknown> = {
            model: a.model ?? VIDEO_MODEL,
            input: parts.length ? [...parts, { type: "text", text: a.prompt }] : a.prompt,
            response_format: { type: "video", delivery: "uri", aspect_ratio: a.aspect_ratio, resolution: a.resolution },
          };
          if (a.continue_from) params.previous_interaction_id = a.continue_from;
          if (a.task) params.generation_config = { video_config: { task: a.task } };
          const it = await interact(params);
          const out = it?.output_video;
          let buf: Buffer | null = null;
          if (out?.uri) buf = await downloadFile(out.uri);
          else if (out?.data) buf = Buffer.from(out.data, "base64");
          if (!buf) return text(noOutput(it, "video"));
          const url = await store(buf, "videos", a.filename, "video", "mp4", "video/mp4");
          return text(`URL: ${url}\ninteraction_id: ${it.id}`);
        } catch (e) {
          return text(errText(e));
        }
      },
    );

    server.registerTool(
      "generate_speech",
      {
        title: "Text to speech (Gemini TTS)",
        description:
          "Text-to-speech with Gemini 3.8 Flash TTS (24kHz WAV). Single speaker: text + voice + optional style. " +
          "Two speakers: turns + speaker_voices. Inline tags: <laugh> <sigh> <gasp> <short pause> <long pause>; " +
          `CAPS = emphasis. Keep text a pure transcript. Voices: ${VOICES}.`,
        inputSchema: z.object({
          text: z.string().optional(),
          voice: z.string().default("Kore"),
          style: z.string().optional().describe('e.g. "excited, fast, Gen Z"'),
          turns: z.array(z.object({ speaker: z.string(), text: z.string(), style: z.string().optional() })).optional(),
          speaker_voices: z.record(z.string(), z.string()).optional().describe('e.g. {"Dorjee":"Puck","Pema":"Kore"}'),
          filename: z.string().optional(),
          model: z.string().optional(),
        }),
      },
      async (a) => {
        try {
          let content: Record<string, unknown>[];
          let speech_config: unknown;
          if (a.turns?.length) {
            content = a.turns.map((t) => ({
              type: "text",
              text: t.text,
              annotations: [{ type: "speech_metadata", speaker: t.speaker, ...(t.style ? { style: t.style } : {}) }],
            }));
            const names = [...new Set(a.turns.map((t) => t.speaker))];
            const defaults = ["Puck", "Kore"];
            speech_config = {
              speakers: names.map((n, i) => ({ speaker: n, voice: a.speaker_voices?.[n] ?? defaults[i % 2] })),
            };
          } else {
            if (!a.text) return text("Pass either `text` or `turns`.");
            content = [{
              type: "text",
              text: a.text,
              ...(a.style ? { annotations: [{ type: "speech_metadata", style: a.style }] } : {}),
            }];
            speech_config = [{ voice: a.voice }];
          }
          const it = await interact({
            model: a.model ?? TTS_MODEL,
            input: [{ type: "user_input", content }],
            response_format: { type: "audio" },
            generation_config: { speech_config },
          });
          if (!it?.output_audio?.data) return text(noOutput(it, "audio"));
          const url = await store(Buffer.from(it.output_audio.data, "base64"), "audio", a.filename, "speech", "wav", "audio/wav");
          return text(`URL: ${url}`);
        } catch (e) {
          return text(errText(e));
        }
      },
    );

    server.registerTool(
      "generate_music",
      {
        title: "Generate music (Lyria)",
        description:
          'Generate music with Lyria 3.5 (MP3). Put length and vocals in the prompt, e.g. "30-second lo-fi Himalayan flute beat, instrumental only".',
        inputSchema: z.object({ prompt: z.string(), filename: z.string().optional(), model: z.string().optional() }),
      },
      async (a) => {
        try {
          const it = await interact({ model: a.model ?? MUSIC_MODEL, input: a.prompt, response_format: { type: "audio" } });
          if (!it?.output_audio?.data) return text(noOutput(it, "music"));
          const url = await store(Buffer.from(it.output_audio.data, "base64"), "music", a.filename, "music", "mp3", "audio/mpeg");
          return text(`URL: ${url}` + (it.output_text ? `\nLyrics/structure:\n${it.output_text}` : ""));
        } catch (e) {
          return text(errText(e));
        }
      },
    );
  },
  { serverInfo: { name: "gemini-studio", version: "1.0.0" } },
);

/** Simple shared-secret auth: ?key=SECRET in the URL, or Authorization: Bearer SECRET. */
async function authed(req: Request): Promise<Response> {
  const secret = process.env.MCP_SECRET;
  if (!secret) return new Response("MCP_SECRET not configured", { status: 500 });
  const url = new URL(req.url);
  const candidates = [
    url.searchParams.get("key"),
    req.headers.get("x-api-key"),
    req.headers.get("x-mcp-key"),
    req.headers.get("authorization")?.replace(/^Bearer\s+/i, ""),
  ].map((v) => v?.trim());
  if (!candidates.includes(secret.trim())) return new Response("Unauthorized", { status: 401 });
  return handler(req);
}

export { authed as GET, authed as POST, authed as DELETE };
