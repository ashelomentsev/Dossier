// OpenAI calls: Whisper transcription, embeddings, label extraction, and dossier
// summaries. This is the only model provider — no Anthropic.

import { Labels } from "./types.ts";

const OPENAI_KEY = Deno.env.get("OPENAI_KEY")!;
const EMBEDDING_MODEL = "text-embedding-3-small"; // 1536 dims — matches the schema
const CHAT_MODEL = "gpt-4o-mini"; // cheap + fast; tasks are simple extraction/summary

/**
 * Transcribe a voice note and normalize it to English. OGG/Opus is sent straight
 * to Whisper's translation endpoint (no ffmpeg), which always returns English —
 * so a note in any language ("Я встретила Юлию") lands as English ("I met
 * Julia"). This keeps names and embeddings in one language space, so the same
 * person matches regardless of the language each note was spoken in.
 */
export async function transcribeToEnglish(audio: Uint8Array): Promise<string> {
  const form = new FormData();
  form.append("file", new Blob([audio], { type: "audio/ogg" }), "voice.ogg");
  form.append("model", "whisper-1");

  const res = await fetch("https://api.openai.com/v1/audio/translations", {
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
  "Extract every salient fact about the person from the note and return a JSON object. " +
  "Capture not just hard facts but also physical description, gender, and the user's " +
  "impressions — these are exactly the cues used later to recognize the person. " +
  "Use lowercase snake_case keys. Only omit a field when the note says nothing about it; " +
  "do not drop a detail just because it isn't in the preferred list. " +
  "Prefer these keys when they apply: name, gender, age, city, location, where_met, job, " +
  "family, appearance, personality, hobby, interests, " +
  "linkedin, twitter, instagram, telegram, facebook, github, website, phone, email. " +
  "Copy URLs, @handles, phone numbers and emails verbatim — never reword or guess them. " +
  "If the note states conflicting facts about the same attribute, the most recent or " +
  'explicitly corrective statement wins (e.g. a later "her name is spelled Sara, not ' +
  'Sarah" overrides an earlier "Sarah"). ' +
  "Use a string for single values and an array of strings for multiple. " +
  'Example note: "I met John in London, a friendly 30 year old software engineer with a ' +
  'beard and glasses, married with 2 kids, plays football and is an Arsenal fan." ' +
  'Example output: {"name":"John","gender":"male","age":"30","city":"London",' +
  '"job":"software engineer","family":"married, 2 kids","appearance":"beard, glasses",' +
  '"personality":"friendly","hobby":"play football","interests":"Arsenal"}';

const URL_RE = /\bhttps?:\/\/[^\s)]+/gi;
const EMAIL_RE = /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/gi;
// Loose phone shape: optional +, then digits/spaces/()-. — digit count is
// validated afterwards so we don't grab years or ages.
const PHONE_RE = /\+?\d[\d\s().-]{7,}\d/g;

function classifyUrl(url: string): string {
  const u = url.toLowerCase();
  if (u.includes("linkedin.com")) return "linkedin";
  if (u.includes("twitter.com") || u.includes("x.com")) return "twitter";
  if (u.includes("instagram.com")) return "instagram";
  if (u.includes("t.me") || u.includes("telegram.me")) return "telegram";
  if (u.includes("facebook.com") || u.includes("fb.com")) return "facebook";
  if (u.includes("github.com")) return "github";
  return "website";
}

/**
 * Pull contact handles (social URLs, email, phone) out of raw text with regexes.
 * LLM extraction tends to paraphrase or truncate URLs, so we capture these
 * verbatim and let them override the model's guesses for the same keys.
 */
export function extractContactHandles(text: string): Labels {
  const acc: Record<string, string[]> = {};
  const add = (key: string, value: string) => {
    const list = acc[key] ?? (acc[key] = []);
    if (!list.includes(value)) list.push(value);
  };

  let residual = text;
  for (const m of text.matchAll(URL_RE)) {
    const url = m[0].replace(/[.,);]+$/, ""); // trim trailing sentence punctuation
    add(classifyUrl(url), url);
  }
  residual = residual.replace(URL_RE, " ");

  for (const m of residual.matchAll(EMAIL_RE)) add("email", m[0]);
  residual = residual.replace(EMAIL_RE, " ");

  for (const m of residual.matchAll(PHONE_RE)) {
    const digits = m[0].replace(/\D/g, "");
    if (digits.length >= 9 && digits.length <= 15) add("phone", m[0].trim());
  }

  const out: Labels = {};
  for (const [key, list] of Object.entries(acc)) {
    out[key] = list.length === 1 ? list[0] : list;
  }
  return out;
}

const CORRECTION_SYSTEM =
  "You maintain a structured JSON profile of a person. You are given the CURRENT profile " +
  "and a correction or addition from the user. Return the FULL updated JSON profile. " +
  "Apply explicit corrections by replacing the affected field — e.g. " +
  '"her name is spelled Sara, not Sarah" sets name to "Sara"; "he moved to Berlin" ' +
  "updates city. Add genuinely new facts as new fields. Keep every existing field " +
  "unchanged unless the correction changes or removes it. If a field is corrected to " +
  'nothing (e.g. "she\'s not actually a doctor"), drop that field. ' +
  "Use lowercase snake_case keys and the same preferred keys as extraction (name, gender, " +
  "age, city, location, where_met, job, family, appearance, personality, hobby, interests, " +
  "linkedin, twitter, instagram, telegram, facebook, github, website, phone, email). " +
  "Copy URLs, @handles, phone numbers and emails verbatim. " +
  "Use a string for single values and an array of strings for multiple.";

/**
 * Apply a correction to an existing profile as a field-level edit, rather than
 * re-deriving labels from the merged note (which would still carry the old
 * value). This is what makes "her name is spelled Sara" actually replace the
 * name instead of accumulating alongside it.
 */
export async function applyCorrection(current: Labels, correction: string): Promise<Labels> {
  const raw = await chat(
    CORRECTION_SYSTEM,
    `Current profile JSON:\n${JSON.stringify(current)}\n\nCorrection: ${correction}`,
    { jsonMode: true },
  );
  let labels: Labels;
  try {
    labels = JSON.parse(raw) as Labels;
  } catch {
    labels = current; // keep what we had rather than lose data on a parse miss
  }
  // Verbatim handles in the correction text override the model's rendering.
  return { ...labels, ...extractContactHandles(correction) };
}

export async function generateLabels(note: string): Promise<Labels> {
  const raw = await chat(LABEL_SYSTEM, `Note: ${note}`, { jsonMode: true });
  let labels: Labels;
  try {
    labels = JSON.parse(raw) as Labels;
  } catch {
    labels = {};
  }
  // Verbatim regex hits win over the model's paraphrase of the same handle.
  return { ...labels, ...extractContactHandles(note) };
}

const STORY_SYSTEM =
  "You are a personal assistant that writes a very short, concise dossier about a person " +
  "from the provided facts and notes. Combine them into a single coherent paragraph. " +
  "Do not invent new facts.";

export async function generateStory(note: string): Promise<string> {
  return chat(STORY_SYSTEM, `Notes: ${note}`, { maxTokens: 400 });
}
