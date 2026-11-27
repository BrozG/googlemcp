import { GoogleGenAI } from "@google/genai";
import { put } from "@vercel/blob";
import sharp from "sharp";
import { readFile, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export const IMAGE_MODEL = process.env.IMAGE_MODEL ?? "gemini-nano-banana-2.1";
export const VIDEO_MODEL = process.env.VIDEO_MODEL ?? "gemini-omni-1.1-flash";
export const TTS_MODEL = process.env.TTS_MODEL ?? "gemini-3.8-flash-tts";
export const MUSIC_MODEL = process.env.MUSIC_MODEL ?? "lyria-3.5";

let _ai: GoogleGenAI | null = null;
export function ai(): GoogleGenAI {
  if (!_ai) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error("GEMINI_API_KEY is not set in Vercel env vars.");
    _ai = new GoogleGenAI({ apiKey });
  }
  return _ai;
}

// Interactions API params are snake_case and the typings move fast; keep this loose.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function interact(params: Record<string, unknown>): Promise<any> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (ai() as any).interactions.create(params);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function slug(name: string | undefined, prefix: string): string {
  const clean = (name ?? "").replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_").replace(/^_+|_+$/g, "");
  const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  return `${clean || prefix}_${stamp}`;
}

/** Upload bytes to Vercel Blob, return the public URL. */
export async function store(
  data: Buffer,
  folder: string,
  name: string | undefined,
  prefix: string,
  ext: string,
  contentType: string,
): Promise<string> {
  const blob = await put(`gemini-studio/${folder}/${slug(name, prefix)}.${ext}`, data, {
    access: "public",
    contentType,
    addRandomSuffix: true,
  });
  return blob.url;
}

/** Fetch a URL (or data: URI) and return an Interactions image part. */
export async function imagePart(src: string) {
  if (src.startsWith("data:")) {
    const m = src.match(/^data:([^;]+);base64,(.*)$/);
    if (!m) throw new Error("Bad data URI");
    return { type: "image", mime_type: m[1], data: m[2] };
  }
  const res = await fetch(src);
  if (!res.ok) throw new Error(`Could not fetch reference image ${src}: ${res.status}`);
  const mime = res.headers.get("content-type")?.split(";")[0] || "image/png";
  const buf = Buffer.from(await res.arrayBuffer());
  return { type: "image", mime_type: mime, data: buf.toString("base64") };
}

/** Small JPEG preview so Claude can see the image without flooding context. */
export async function preview(data: Buffer): Promise<string> {
  const out = await sharp(data).resize(640, 640, { fit: "inside" }).jpeg({ quality: 78 }).toBuffer();
  return out.toString("base64");
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function stateOf(f: any): string {
  return String(f?.state?.name ?? f?.state ?? "").toUpperCase();
}

export async function waitActive(name: string, timeoutMs = 240_000) {
  const start = Date.now();
  for (;;) {
    const st = stateOf(await ai().files.get({ name }));
    if (st.includes("ACTIVE")) return;
    if (st.includes("FAILED")) throw new Error(`File ${name} failed processing`);
    if (Date.now() - start > timeoutMs) throw new Error(`File ${name} not ready in time`);
    await sleep(4000);
  }
}

/** Upload a remote video (by URL) to Gemini Files, return its file URI. */
export async function uploadVideoFromUrl(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not fetch video ${url}: ${res.status}`);
  const blob = await res.blob();
  const file = await ai().files.upload({ file: blob, config: { mimeType: blob.type || "video/mp4" } });
  if (!file.name) throw new Error("Upload returned no file name");
  await waitActive(file.name);
  const ready = await ai().files.get({ name: file.name });
  if (!ready.uri) throw new Error("Uploaded file has no URI");
  return ready.uri;
}

/** Download a Gemini file URI to memory via /tmp. */
export async function downloadFile(uri: string): Promise<Buffer> {
  const id = uri.replace(/\/+$/, "").split("/").pop()!;
  await waitActive(`files/${id}`);
  const path = join(tmpdir(), `${id}.mp4`);
  await ai().files.download({ file: `files/${id}`, downloadPath: path });
  const buf = await readFile(path);
  await unlink(path).catch(() => {});
  return buf;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function noOutput(it: any, kind: string): string {
  return `No ${kind} returned (possibly blocked by safety filters). Model said: ${it?.output_text ?? "nothing"}`;
}

export function errText(e: unknown): string {
  return e instanceof Error ? `Error: ${e.name}: ${e.message}` : `Error: ${String(e)}`;
}
