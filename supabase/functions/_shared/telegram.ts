// Thin wrapper over the Telegram Bot API.

const TOKEN = Deno.env.get("TELEGRAM_TOKEN")!;
const WEBHOOK_SECRET = Deno.env.get("TELEGRAM_WEBHOOK_SECRET");
const BASE_URL = `https://api.telegram.org/bot${TOKEN}`;

/**
 * Telegram echoes the secret configured via setWebhook in this header. We reject
 * any request that doesn't match, so the public function URL can't be driven by
 * a stranger. If no secret is configured, verification is skipped.
 */
export function verifyTelegramSecret(req: Request): boolean {
  if (!WEBHOOK_SECRET) return true;
  return req.headers.get("X-Telegram-Bot-Api-Secret-Token") === WEBHOOK_SECRET;
}

interface InlineButton {
  text: string;
  callback_data: string;
}

export async function sendMessage(
  chatId: number,
  text: string,
  buttons?: InlineButton[],
): Promise<void> {
  const body: Record<string, unknown> = {
    chat_id: chatId,
    text,
    parse_mode: "Markdown",
  };
  if (buttons?.length) {
    body.reply_markup = { inline_keyboard: [buttons] };
  }

  const res = await fetch(`${BASE_URL}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    console.error("sendMessage failed", res.status, await res.text());
  }
}

/** Resolve a Telegram file_id to a temporary download path. */
async function getFilePath(fileId: string): Promise<string> {
  const res = await fetch(`${BASE_URL}/getFile?file_id=${fileId}`);
  const info = await res.json();
  if (!info.ok) throw new Error(`getFile failed: ${JSON.stringify(info)}`);
  return info.result.file_path;
}

/** Download a voice note's bytes. Telegram voice notes are OGG/Opus. */
export async function downloadVoice(fileId: string): Promise<Uint8Array> {
  const filePath = await getFilePath(fileId);
  const res = await fetch(
    `https://api.telegram.org/file/bot${TOKEN}/${filePath}`,
  );
  if (!res.ok) throw new Error(`file download failed: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}
