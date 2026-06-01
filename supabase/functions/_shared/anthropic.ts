// Anthropic calls: extract structured labels from a note, and write a short story.
// The original app asked Claude for XML tags and parsed them; here we ask for JSON,
// which is more robust to parse in Deno and maps cleanly onto the `labels` jsonb.

import { Labels } from "./types.ts";

const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY")!;
const MODEL = "claude-haiku-4-5"; // fast + cheap; tasks are simple extraction/summary

async function complete(system: string, user: string, maxTokens = 512): Promise<string> {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: maxTokens,
      system,
      messages: [{ role: "user", content: user }],
    }),
  });
  if (!res.ok) {
    throw new Error(`Anthropic failed: ${res.status} ${await res.text()}`);
  }
  const data = await res.json();
  return (data.content?.[0]?.text ?? "").trim();
}

const LABEL_SYSTEM =
  "You are a personal assistant that stores information about people the user meets. " +
  "Extract key facts about the person from the note and return ONLY a JSON object — no prose, " +
  "no code fences. Use lowercase snake_case keys. Omit any field you don't have. " +
  "Prefer these keys when they apply: name, age, city, job, family, hobby, interests, location. " +
  "Use a string for single values and an array of strings for multiple. " +
  'Example note: "I met John in London, he is a 30 year old software engineer, married with 2 kids, ' +
  'plays football and is an Arsenal fan." ' +
  'Example output: {"name":"John","age":"30","city":"London","job":"software engineer",' +
  '"family":"married, 2 kids","hobby":"play football","interests":"Arsenal"}';

/** Pull JSON out of a model response, tolerating stray prose or code fences. */
function parseLabels(raw: string): Labels {
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) return {};
  try {
    return JSON.parse(match[0]) as Labels;
  } catch {
    return {};
  }
}

export async function generateLabels(note: string): Promise<Labels> {
  return parseLabels(await complete(LABEL_SYSTEM, `Note: ${note}`));
}

const STORY_SYSTEM =
  "You are a personal assistant that writes a very short, concise dossier about a person " +
  "from the provided facts and notes. Combine them into a single coherent paragraph. " +
  "Do not invent new facts.";

export async function generateStory(note: string): Promise<string> {
  return complete(STORY_SYSTEM, `Notes: ${note}`, 400);
}
