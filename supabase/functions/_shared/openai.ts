// OpenAI calls: Whisper transcription, embeddings, label extraction, and dossier
// summaries. This is the only model provider — no Anthropic.

import { Labels } from "./types.ts";

const OPENAI_KEY = Deno.env.get("OPENAI_KEY")!;
const EMBEDDING_MODEL = "text-embedding-3-small"; // 1536 dims — matches the schema
const CHAT_MODEL = "gpt-4o-mini"; // cheap + fast; tasks are simple extraction/summary

/** Transcribe a voice note. OGG/Opus is sent straight to Whisper (no ffmpeg). */
export async function transcribe(audio: Uint8Array): Promise<string> {
  const form = new FormData();
  form.append("file", new Blob([audio], { type: "audio/ogg" }), "voice.ogg");
  form.append("model", "whisper-1");

  const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_KEY}` },
    body: form,
  });
  if (!res.ok) {
    throw new Error(`Whisper failed: ${res.status} ${await res.text()}`);
  }
  const data = await res.json();
  return (data.text ?? "").trim();
}

/** Embed text with text-embedding-3-small. */
export async function embed(text: string): Promise<number[]> {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: text }),
  });
  if (!res.ok) {
    throw new Error(`Embeddings failed: ${res.status} ${await res.text()}`);
  }
  const data = await res.json();
  return data.data[0].embedding;
}

/** One chat completion turn. Pass jsonMode for guaranteed-parseable JSON output. */
async function chat(
  system: string,
  user: string,
  opts: { jsonMode?: boolean; maxTokens?: number } = {},
): Promise<string> {
  const body: Record<string, unknown> = {
    model: CHAT_MODEL,
    max_tokens: opts.maxTokens ?? 512,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
  };
  if (opts.jsonMode) body.response_format = { type: "json_object" };

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Chat failed: ${res.status} ${await res.text()}`);
  }
  const data = await res.json();
  return (data.choices?.[0]?.message?.content ?? "").trim();
}

const LABEL_SYSTEM =
  "You are a personal assistant that stores information about people the user meets. " +
  "Extract key facts about the person from the note and return a JSON object. " +
  "Use lowercase snake_case keys. Omit any field you don't have. " +
  "Prefer these keys when they apply: name, age, city, job, family, hobby, interests, location. " +
  "Use a string for single values and an array of strings for multiple. " +
  'Example note: "I met John in London, he is a 30 year old software engineer, married with 2 kids, ' +
  'plays football and is an Arsenal fan." ' +
  'Example output: {"name":"John","age":"30","city":"London","job":"software engineer",' +
  '"family":"married, 2 kids","hobby":"play football","interests":"Arsenal"}';

export async function generateLabels(note: string): Promise<Labels> {
  const raw = await chat(LABEL_SYSTEM, `Note: ${note}`, { jsonMode: true });
  try {
    return JSON.parse(raw) as Labels;
  } catch {
    return {};
  }
}

const STORY_SYSTEM =
  "You are a personal assistant that writes a very short, concise dossier about a person " +
  "from the provided facts and notes. Combine them into a single coherent paragraph. " +
  "Do not invent new facts.";

export async function generateStory(note: string): Promise<string> {
  return chat(STORY_SYSTEM, `Notes: ${note}`, { maxTokens: 400 });
}
