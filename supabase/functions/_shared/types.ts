// Minimal shapes for the slices of the Telegram Bot API we consume, plus
// domain types shared across the function.

export interface TelegramChat {
  id: number;
}

export interface TelegramVoice {
  file_id: string;
  duration?: number;
  mime_type?: string;
}

// A formatting span on a message. We care about `text_link`, whose `url` we use
// to smuggle a record id into our confirmations (recovered on reply).
export interface TelegramMessageEntity {
  type: string;
  offset: number;
  length: number;
  url?: string;
}

export interface TelegramMessage {
  message_id: number;
  chat: TelegramChat;
  text?: string;
  voice?: TelegramVoice;
  entities?: TelegramMessageEntity[];
  // Present when the user replies to a message; carries the full original
  // message (including its entities), which is how a reply round-trips a record id.
  reply_to_message?: TelegramMessage;
}

export interface TelegramCallbackQuery {
  id: string;
  data?: string;
  message?: TelegramMessage;
}

export interface TelegramUpdate {
  update_id: number;
  message?: TelegramMessage;
  callback_query?: TelegramCallbackQuery;
}

// Structured fields extracted from a note, e.g. { name: "Sarah", city: "Tirana" }.
export type Labels = Record<string, string | string[]>;

// A row from the `people` table / match_person RPC.
export interface Person {
  id: string;
  name: string | null;
  note: string;
  labels: Labels;
  similarity?: number;
}

export interface UserState {
  user_id: number;
  storage_mode: "supabase" | "vault";
  search_mode: boolean;
}
