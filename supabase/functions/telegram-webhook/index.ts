// Dossier — Telegram webhook (Phase 1, Supabase-canonical).
//
// Flow: voice note -> Whisper transcript -> embed -> nearest-person match.
//   - capture mode: update the matched person, or create a new one
//   - search mode (/search): return a dossier on the matched person
//
// Processing is synchronous: we do the work, then ack Telegram. The chain is
// mostly network I/O, so it stays within Edge Function limits.

import {
  answerCallbackQuery,
  deleteMessage,
  downloadVoice,
  sendMessage,
  verifyTelegramSecret,
} from "../_shared/telegram.ts";
import {
  applyCorrection,
  embed,
  generateLabels,
  generateStory,
  transcribeToEnglish,
} from "../_shared/openai.ts";
import { formatLabels, personName } from "../_shared/labels.ts";
import {
  deletePerson,
  ensureUser,
  getPerson,
  insertPerson,
  matchPerson,
  searchPeople,
  setSearchMode,
  updatePerson,
} from "../_shared/db.ts";
import {
  Person,
  TelegramCallbackQuery,
  TelegramMessage,
  TelegramUpdate,
} from "../_shared/types.ts";

const WELCOME = `*Welcome to DOSSIER — your Connections Concierge!*
_Here's how it works:_

1. Send a voice note describing someone you met, e.g. "I just met Sarah at a hackathon, she's a data analyst from Albania."
2. To add more, just say: "Met Sarah again — the data analyst. She has a cute dog named Winnie."
3. *Reply* to my confirmation to attach corrections — paste a LinkedIn/socials URL, phone, or email, or send a voice note with clarifications.
4. Use /search, then describe a person to get their dossier.

*🦸 Enjoy your augmented memory like a super-human!*`;

// Carrier for record ids inside confirmation messages. The domain need not
// resolve — it only exists so Telegram stores a `text_link` entity whose URL we
// recover from `reply_to_message` when the user replies to correct the record.
const RECORD_LINK_BASE = "https://dossier.app/p/";
const RECORD_ID_RE = /\/p\/([0-9a-f-]{36})/i;

/**
 * A neutral record-id tag rather than a call-to-action: the visible text is just
 * "Record ID" (not meant to be clicked), while the id rides invisibly in the URL
 * so a Telegram reply to this message addresses exactly this record.
 */
function replyHint(personId: string): string {
  return `\n\n[Record ID](${RECORD_LINK_BASE}${personId})`;
}

// callback_data prefixes. `<prefix><uuid>` stays well under Telegram's 64-byte
// callback_data limit. EDIT starts a correction; SHOW opens a dossier from the
// recall shortlist; DEL asks to delete, DELYES/DELNO resolve that confirmation.
// Prefixes are distinct strings (delyes:/delno: don't start with "del:"), so the
// dispatcher's startsWith checks don't collide.
const EDIT_PREFIX = "edit:";
const SHOW_PREFIX = "show:";
const DEL_PREFIX = "del:";
const DELYES_PREFIX = "delyes:";
const DELNO_PREFIX = "delno:";

/** The inline button shown on every confirmation to start a correction. */
function editButton(personId: string) {
  return [{ text: "✍️ Add changes", callback_data: `${EDIT_PREFIX}${personId}` }];
}

/** Buttons shown under a dossier: start a correction, or delete the record. */
function dossierButtons(personId: string) {
  return [
    { text: "✍️ Add changes", callback_data: `${EDIT_PREFIX}${personId}` },
    { text: "🗑 Delete", callback_data: `${DEL_PREFIX}${personId}` },
  ];
}

/** Short, disambiguating label for a shortlist button: "Tilly · Revolut, London". */
function personButtonLabel(p: Person): string {
  const name = p.name ?? personName(p.labels) ?? "Unknown";
  const hints: string[] = [];
  for (const k of ["job", "city", "where_met", "location"]) {
    const v = p.labels?.[k];
    if (v) hints.push(Array.isArray(v) ? String(v[0]) : String(v));
    if (hints.length >= 2) break;
  }
  const label = hints.length ? `${name} · ${hints.join(", ")}` : name;
  return label.length > 60 ? `${label.slice(0, 57)}…` : label;
}

/**
 * Render a person's dossier (labels + a short generated summary), with the hidden
 * record-id link plus Add-changes / Delete buttons so the user can act on it.
 */
async function sendDossier(chatId: number, person: Person): Promise<void> {
  const story = await generateStory(person.note);
  const header = person.name ? `Here's what I have on *${person.name}*:` : "Found a match:";
  const body = `${header}\n\n${formatLabels(person.labels)}\n\n${story}`.trim();
  await sendMessage(chatId, `${body}${replyHint(person.id)}`, {
    buttons: dossierButtons(person.id),
  });
}

/** Recover the record id from a replied-to confirmation (link entity, or text). */
function extractRecordId(msg?: TelegramMessage): string | null {
  if (!msg) return null;
  for (const e of msg.entities ?? []) {
    if (e.type === "text_link" && e.url) {
      const m = e.url.match(RECORD_ID_RE);
      if (m) return m[1];
    }
  }
  const m = (msg.text ?? "").match(RECORD_ID_RE);
  return m ? m[1] : null;
}

/**
 * Heuristic recall detector. Runs on the English-normalized transcript, so it
 * works regardless of the spoken language. Catches explicit questions ("What's
 * the name of Oriona's owner?") so they recall a person instead of being saved
 * as a new record, even when the user forgot to send /search first.
 */
function looksLikeRecall(text: string): boolean {
  const t = text.trim().toLowerCase();
  if (t.endsWith("?")) return true;
  return /^(who|whos|who's|what|whats|what's|where|which|when|how|do|does|did|is|are|was|were|can|could|tell me|remind me|find|search|look up|show me|recall)\b/
    .test(t);
}

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
  console.log("handleVoice", { chatId, searchMode: user.search_mode });

  const audio = await downloadVoice(message.voice!.file_id);
  const transcription = await transcribeToEnglish(audio);
  console.log("transcription", { chatId, text: transcription });
  if (!transcription) {
    await sendMessage(chatId, "I couldn't make out any speech — please try again.");
    return;
  }

  const embedding = await embed(transcription);
  // Extract labels up front so the name can drive matching (name-aware match
  // recognizes the same person even when notes describe them differently).
  const labels = await generateLabels(transcription);
  const name = personName(labels);

  // Recall if the user is in /search mode OR the note reads like a question.
  const isRecallQuestion = looksLikeRecall(transcription);
  const wantRecall = user.search_mode || isRecallQuestion;

  // Recall mode: never write. Uses a permissive floor (searchPeople) rather than
  // the strict capture threshold — a short query rarely reaches 0.7 against a full
  // note, so reusing the capture match would find no one. Returns up to 3 so the
  // user can disambiguate when several people are close.
  if (wantRecall) {
    if (user.search_mode) await setSearchMode(chatId, false);
    const matches = await searchPeople(chatId, embedding, name, 3);
    console.log("branch: search", { chatId, queryName: name, found: matches.length });
    if (matches.length === 0) {
      await sendMessage(chatId, "🤷 No one in your memory matches that description yet.");
    } else if (matches.length === 1) {
      await sendDossier(chatId, matches[0]);
    } else {
      // Ambiguous — let the user pick the one they meant from a tappable shortlist.
      const buttons = matches.map((p) => ({
        text: personButtonLabel(p),
        callback_data: `${SHOW_PREFIX}${p.id}`,
      }));
      await sendMessage(chatId, "🔎 I found a few people — which one did you mean?", { buttons });
    }
    return;
  }

  // Capture mode: update the matched person, or create a new one. Strict
  // threshold (matchPerson) so a new note isn't merged into the wrong record.
  const match = await matchPerson(chatId, embedding, name);
  console.log("matchResult", {
    chatId,
    queryName: name,
    matched: !!match,
    matchId: match?.id ?? null,
  });
  if (match) {
    console.log("branch: capture/update", { chatId, matchId: match.id });
    // Re-extract from the full merged note so labels reflect all accumulated facts.
    const mergedNote = `${match.note}\n\n${transcription}`;
    const mergedLabels = await generateLabels(mergedNote);
    const mergedName = personName(mergedLabels) ?? match.name;
    const mergedEmbedding = await embed(mergedNote);
    await updatePerson(match.id, chatId, {
      name: mergedName,
      note: mergedNote,
      labels: mergedLabels,
      embedding: mergedEmbedding,
    });
    await sendMessage(
      chatId,
      `✍️ Updated ${mergedName ?? "an existing person"} with the new details:\n\n${
        formatLabels(mergedLabels)
      }${replyHint(match.id)}`,
      { buttons: editButton(match.id) },
    );
  } else {
    console.log("branch: capture/insert", { chatId });
    // Reuse the labels/name already extracted above for matching.
    const newId = await insertPerson(chatId, { name, note: transcription, labels, embedding });
    await sendMessage(
      chatId,
      `🙅 No similar person found — saving a new record:\n\n${formatLabels(labels)}${
        replyHint(newId)
      }`,
      { buttons: editButton(newId) },
    );
  }
}

/**
 * A reply to one of our confirmations: merge the correction (typed text or a
 * voice clarification) into the addressed record. Targets the record by id, so
 * it never depends on the fuzzy match — and re-extracts labels so a pasted
 * LinkedIn URL or phone number lands in a structured field.
 */
async function handleCorrection(message: TelegramMessage, personId: string): Promise<void> {
  const chatId = message.chat.id;
  await ensureUser(chatId);

  let addition: string;
  if (message.voice) {
    const audio = await downloadVoice(message.voice.file_id);
    addition = await transcribeToEnglish(audio);
  } else {
    addition = (message.text ?? "").trim();
  }
  if (!addition) {
    await sendMessage(chatId, "I couldn't read that correction — send text or a voice note.");
    return;
  }

  const person = await getPerson(personId, chatId);
  if (!person) {
    await sendMessage(chatId, "I couldn't find that record to update.");
    return;
  }
  console.log("branch: correction", { chatId, personId });

  const mergedNote = `${person.note}\n\n${addition}`;
  // Apply as a field-level edit to the existing labels (so "her name is spelled
  // Sara" replaces the name) instead of re-deriving from the merged note, which
  // would still carry the old value. The note keeps the full history for the
  // embedding and the dossier.
  const mergedLabels = await applyCorrection(person.labels, addition);
  const mergedName = personName(mergedLabels) ?? person.name;
  const mergedEmbedding = await embed(mergedNote);
  await updatePerson(person.id, chatId, {
    name: mergedName,
    note: mergedNote,
    labels: mergedLabels,
    embedding: mergedEmbedding,
  });
  await sendMessage(
    chatId,
    `✍️ Updated ${mergedName ?? "the record"}:\n\n${formatLabels(mergedLabels)}${
      replyHint(person.id)
    }`,
    { buttons: editButton(person.id) },
  );
}

/** Route a tapped inline button to the right action by its callback_data prefix. */
async function handleCallbackQuery(cb: TelegramCallbackQuery): Promise<void> {
  await answerCallbackQuery(cb.id); // stop the button spinner regardless
  const chatId = cb.message?.chat.id;
  const data = cb.data ?? "";
  if (!chatId) return;

  if (data.startsWith(EDIT_PREFIX)) {
    await startCorrection(cb, chatId, data.slice(EDIT_PREFIX.length));
  } else if (data.startsWith(SHOW_PREFIX)) {
    await showDossier(chatId, data.slice(SHOW_PREFIX.length));
  } else if (data.startsWith(DELYES_PREFIX)) {
    await confirmDelete(cb, chatId, data.slice(DELYES_PREFIX.length));
  } else if (data.startsWith(DELNO_PREFIX)) {
    await cancelDelete(cb, chatId);
  } else if (data.startsWith(DEL_PREFIX)) {
    await askDelete(chatId, data.slice(DEL_PREFIX.length));
  }
}

/** Delete button: ask for confirmation first — deletion is irreversible. */
async function askDelete(chatId: number, personId: string): Promise<void> {
  const person = await getPerson(personId, chatId);
  if (!person) {
    await sendMessage(chatId, "I couldn't find that record.");
    return;
  }
  await sendMessage(
    chatId,
    `🗑 Delete *${person.name ?? "this person"}*? This can't be undone.`,
    {
      buttons: [
        { text: "✅ Yes, delete", callback_data: `${DELYES_PREFIX}${personId}` },
        { text: "✖️ Cancel", callback_data: `${DELNO_PREFIX}${personId}` },
      ],
    },
  );
}

/** Confirmed delete: remove the record and the confirmation prompt. */
async function confirmDelete(
  cb: TelegramCallbackQuery,
  chatId: number,
  personId: string,
): Promise<void> {
  const person = await getPerson(personId, chatId);
  if (cb.message?.message_id) await deleteMessage(chatId, cb.message.message_id);
  if (!person) {
    await sendMessage(chatId, "That record is already gone.");
    return;
  }
  await deletePerson(personId, chatId);
  console.log("branch: delete", { chatId, personId });
  await sendMessage(chatId, `🗑 Deleted *${person.name ?? "the record"}*.`);
}

/** Cancelled delete: just dismiss the confirmation prompt. */
async function cancelDelete(cb: TelegramCallbackQuery, chatId: number): Promise<void> {
  if (cb.message?.message_id) await deleteMessage(chatId, cb.message.message_id);
}

/** Recall-shortlist pick: open the chosen person's dossier. */
async function showDossier(chatId: number, personId: string): Promise<void> {
  const person = await getPerson(personId, chatId);
  if (!person) {
    await sendMessage(chatId, "I couldn't find that record.");
    return;
  }
  console.log("branch: show-dossier", { chatId, personId });
  await sendDossier(chatId, person);
}

/**
 * "Add changes" button tap. An inline button can't open a reply box itself, so
 * we swap the confirmation for a force-reply prompt: delete the original and
 * re-post the record (carrying its id) with force_reply. The user's reply then
 * flows through the normal reply path (handleCorrection).
 */
async function startCorrection(
  cb: TelegramCallbackQuery,
  chatId: number,
  personId: string,
): Promise<void> {
  const person = await getPerson(personId, chatId);
  if (!person) {
    await sendMessage(chatId, "I couldn't find that record to edit.");
    return;
  }
  console.log("branch: edit-button", { chatId, personId });

  // Swap the confirmation (inline button) for a force-reply prompt; the two
  // reply_markup kinds can't coexist on one message, hence delete + re-post.
  if (cb.message?.message_id) {
    await deleteMessage(chatId, cb.message.message_id);
  }
  await sendMessage(
    chatId,
    `✍️ Send your correction for *${person.name ?? "this person"}* — type it or record a ` +
      `voice note (e.g. "her name is spelled Sara", or paste a LinkedIn URL):\n\n` +
      `${formatLabels(person.labels)}${replyHint(person.id)}`,
    { forceReply: true, placeholder: "Type or record your correction…" },
  );
}

async function handleUpdate(update: TelegramUpdate): Promise<void> {
  if (update.callback_query) {
    await handleCallbackQuery(update.callback_query);
    return;
  }

  const message = update.message;
  if (!message) return;

  // A reply to one of our record confirmations is a correction to that record.
  // Check this first: such a reply may be a voice note or text, and either way
  // it should edit the addressed record rather than start a new capture.
  const replyId = extractRecordId(message.reply_to_message);
  if (replyId && (message.voice || message.text)) {
    await handleCorrection(message, replyId);
    return;
  }

  if (message.voice) {
    await handleVoice(message);
  } else if (message.text) {
    await handleText(message);
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
