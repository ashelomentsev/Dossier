// OpenAI calls: Whisper transcription and embeddings.

const OPENAI_KEY = Deno.env.get("OPENAI_KEY")!;
const EMBEDDING_MODEL = "text-embedding-3-small"; // 1536 dims — matches the schema

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
