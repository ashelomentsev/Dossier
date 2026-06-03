# Dossier — Connections Concierge

An AI-augmented memory for the people you meet. Send the Telegram bot a voice note about someone,
and Dossier transcribes it, extracts structured facts, and remembers them. Later, describe a person
and Dossier hands you back a dossier.

> _"Enjoy your augmented memory like a super-human."_

Built on **Supabase Edge Functions** (Deno/TypeScript) with **Postgres + pgvector** as the per-user
memory store.

## How it works

1. **Capture** — You send a voice note ("I met Sarah at a hackathon, she's a data analyst from
   Albania").
2. **Transcribe** — The OGG voice note goes straight to OpenAI Whisper.
3. **Match** — The transcript is embedded (`text-embedding-3-small`) and compared against people you
   already know via pgvector.
4. **Remember** — If it's someone you already know (cosine similarity above `SIM_THRESHOLD`), the
   new facts are **merged** into their record. Otherwise a **new** person is created. GPT extracts
   the structured fields.
5. **Recall** — Send `/search`, then describe a person; Dossier finds the closest match and writes a
   short dossier.
6. **Correct** — _Reply_ to a confirmation message to attach edits to that exact record (no fuzzy
   matching): paste a LinkedIn/socials URL, phone, or email, or send a voice note with
   clarifications. URLs, handles, phones and emails are captured verbatim into structured fields.
   The record id rides invisibly in the confirmation's link and round-trips back via the reply.

### Telegram commands

| Command   | Description                                           |
| --------- | ----------------------------------------------------- |
| `/start`  | Show the welcome message.                             |
| `/search` | Next voice note is treated as a recall query.         |
| _(voice)_ | Add a new person or update an existing one.           |
| _(reply)_ | Correct/extend the replied-to record (text or voice). |

## Architecture

```
Telegram ──webhook──▶ Edge Function (Deno/TS)
                          ├─▶ OpenAI Whisper        (transcribe)
                          ├─▶ OpenAI Embeddings     (text-embedding-3-small)
                          ├─▶ OpenAI Chat           (labels + dossier, gpt-4o-mini)
                          └─▶ Postgres + pgvector   (per-user memory)
```

Everything is **multi-tenant**: each Telegram user gets isolated data, keyed by their chat id and
protected by Row Level Security.

### Layout

| Path                                           | Responsibility                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| `supabase/migrations/0001_init.sql`            | Schema: `users`, `people`, the `match_person` search function, RLS |
| `supabase/functions/telegram-webhook/index.ts` | Webhook entry point and routing                                    |
| `supabase/functions/_shared/telegram.ts`       | Telegram Bot API (send, download, secret check)                    |
| `supabase/functions/_shared/openai.ts`         | Whisper, embeddings, label extraction, dossier summaries           |
| `supabase/functions/_shared/labels.ts`         | Format labels as Telegram Markdown                                 |
| `supabase/functions/_shared/db.ts`             | Supabase data access (match / insert / update)                     |

## Setup

### Prerequisites

- [Supabase CLI](https://supabase.com/docs/guides/cli)
- A Supabase project
- A Telegram bot token ([@BotFather](https://t.me/BotFather))
- An OpenAI API key

### 1. Link and apply the schema

```bash
supabase link --project-ref <your-project-ref>
supabase db push
```

### 2. Configure secrets

```bash
cp .env.example .env   # fill in the values
supabase secrets set --env-file ./.env
```

`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are injected automatically — you don't set those.

### 3. Deploy

```bash
supabase functions deploy telegram-webhook   # verify_jwt is off via config.toml
```

### 4. Register the webhook

```bash
curl "https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook" \
  -d "url=https://<project-ref>.supabase.co/functions/v1/telegram-webhook" \
  -d "secret_token=<TELEGRAM_WEBHOOK_SECRET>"
```

### Local development

```bash
deno task serve   # runs the function locally with ./.env
```

## Roadmap

- **Phase 1 (this code)** — Multi-tenant bot; Supabase is the source of truth.
- **Phase 2 — Obsidian** — Connect your own GitHub-backed Obsidian vault. Your Markdown vault
  becomes canonical (one note per person, frontmatter labels, wikilink graph) and Supabase/pgvector
  becomes a rebuildable search index. The schema already carries the `storage_mode` / `vault_repo` /
  `gh_installation_id` fields for this.

## Security & privacy notes

- **No secrets in source.** All keys come from environment/Supabase secrets. Rotate any key that
  ever lands in git history.
- **The webhook is authenticated** via Telegram's secret-token header.
- **You're storing personal data about third parties.** Plan for export and deletion paths before
  running this for real users; per-user data is isolated via RLS from the start.

## License

No license file is currently included. Add one before distributing.
