// Dossier — Telegram webhook (Phase 1, Supabase-canonical).
//
// Flow: voice note -> Whisper transcript -> embed -> nearest-person match.
//   - capture mode: update the matched person, or create a new one
//   - search mode (/search): return a dossier on the matched person
//
// Processing is synchronous: we do the work, then ack Telegram. The chain is
// mostly network I/O, so it stays within Edge Function limits.

import {
  downloadVoice,
  sendMessage,
  verifyTelegramSecret,
} from "../_shared/telegram.ts";
import { embed, generateLabels, generateStory, transcribe } from "../_shared/openai.ts";
import { formatLabels, personName } from "../_shared/labels.ts";
import {
  ensureUser,
  insertPerson,
  matchPerson,
  setSearchMode,
  updatePerson,
} from "../_shared/db.ts";
import {
  TelegramCallbackQuery,
  TelegramMessage,
  TelegramUpdate,
} from "../_shared/types.ts";

const WELCOME = `*Welcome to DOSSIER — your Connections Concierge!*
_Here's how it works:_

1. Send a voice note describing someone you met, e.g. "I just met Sarah at a hackathon, she's a data analyst from Albania."
2. To add more, just say: "Met Sarah again — the data analyst. She has a cute dog named Winnie."
3. Use /search, then describe a person to get their dossier.

*🦸 Enjoy your augmented memory like a super-human!*`;

async function handleText(message: TelegramMessage): Promise<void> {
  const chatId = message.chat.id;
  const text = (message.text ?? "").trim();

  if (text === "/start") {
    await ensureUser(chatId);
    await sendMessage(chatId, WELCOME);
  } else if (text === "/search") {
    await setSearchMode(chatId, true);
    await sendMessage(chatId, "🔎 Search mode on. Send a voice note describing the person.");
  } else {
    await sendMessage(
      chatId,
      "Send a voice note describing a person, or use /search to recall someone.",
    );
  }
}

async function handleVoice(message: TelegramMessage): Promise<void> {
  const chatId = message.chat.id;
  const user = await ensureUser(chatId);

  const audio = await downloadVoice(message.voice!.file_id);
  const transcription = await transcribe(audio);
  if (!transcription) {
    await sendMessage(chatId, "I couldn't make out any speech — please try again.");
    return;
  }

  const embedding = await embed(transcription);
  const match = await matchPerson(chatId, embedding);

  // Search mode: recall an existing person, never write.
  if (user.search_mode) {
    await setSearchMode(chatId, false);
    if (match) {
      const story = await generateStory(match.note);
      const header = match.name ? `Here's what I have on *${match.name}*:` : "Found a match:";
      const labels = formatLabels(match.labels);
      await sendMessage(chatId, `${header}\n\n${labels}\n\n${story}`.trim());
    } else {
      await sendMessage(chatId, "🤷 No one in your memory matches that description yet.");
    }
    return;
  }

  // Capture mode: update the matched person, or create a new one.
  if (match) {
    const mergedNote = `${match.note}\n\n${transcription}`;
    const labels = await generateLabels(mergedNote);
    const name = personName(labels) ?? match.name;
    const mergedEmbedding = await embed(mergedNote);
    await updatePerson(match.id, {
      name,
      note: mergedNote,
      labels,
      embedding: mergedEmbedding,
    });
    await sendMessage(
      chatId,
      `✍️ Updated ${name ?? "an existing person"} with the new details:\n\n${formatLabels(labels)}`,
    );
  } else {
    const labels = await generateLabels(transcription);
    const name = personName(labels);
    await insertPerson(chatId, { name, note: transcription, labels, embedding });
    await sendMessage(
      chatId,
      `🙅 No similar person found — saving a new record:\n\n${formatLabels(labels)}`,
    );
  }
}

async function handleCallbackQuery(cb: TelegramCallbackQuery): Promise<void> {
  const chatId = cb.message?.chat.id;
  if (!chatId || !cb.data) return;
  // Reserved for the "full story" button (Phase 2). No-op for now.
}

async function handleUpdate(update: TelegramUpdate): Promise<void> {
  if (update.callback_query) {
    await handleCallbackQuery(update.callback_query);
  } else if (update.message?.voice) {
    await handleVoice(update.message);
  } else if (update.message?.text) {
    await handleText(update.message);
  }
}

Deno.serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }
  if (!verifyTelegramSecret(req)) {
    return new Response("Unauthorized", { status: 401 });
  }

  try {
    const update = (await req.json()) as TelegramUpdate;
    await handleUpdate(update);
  } catch (err) {
    // Swallow errors so Telegram doesn't retry-storm; log for debugging.
    console.error("handler error", err);
  }
  return new Response("ok", { status: 200 });
});
