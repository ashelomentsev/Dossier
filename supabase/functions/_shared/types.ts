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

export interface TelegramMessage {
  message_id: number;
  chat: TelegramChat;
  text?: string;
  voice?: TelegramVoice;
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
