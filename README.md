# Dossier — Connections Concierge

An AI-augmented memory for the people you meet. Send the Telegram bot a voice
note describing someone you just met, and Dossier transcribes it, extracts
structured facts, and stores them in a personal, searchable knowledge base.
Later, describe a person and Dossier returns a short dossier on them.

> *"Enjoy your augmented memory like a super-human."*

## How it works

1. **Capture** — You send a voice note ("I met Sarah at a hackathon, she's a
   data analyst from Albania").
2. **Transcribe** — The audio is converted to WAV and transcribed with OpenAI
   Whisper.
3. **Extract** — Claude extracts structured fields (`name`, `age`, `city`,
   `job`, `family`, `hobby`, `interests`, …) from the transcription.
4. **Store / update** — The note is embedded and saved in a per-user
   [FAISS](https://github.com/facebookresearch/faiss) vector store. If a note
   is similar enough to an existing person (cosine distance below
   `SIM_THRESHOLD`), that person's record is updated instead of creating a new
   one.
5. **Search** — Use `/search`, then send a voice note describing someone. The
   bot retrieves the closest match and generates a concise summary.

### Telegram commands

| Command   | Description                                            |
| --------- | ------------------------------------------------------ |
| `/start`  | Show the welcome message and usage instructions.       |
| `/search` | Enter search mode; the next voice note is a query.     |
| *(voice)* | Add a new person or update an existing one.            |

## Architecture

| File                 | Responsibility                                             |
| -------------------- | --------------------------------------------------------- |
| `app.py`             | Flask app, Telegram webhook, voice handling, label/story generation |
| `faiss_retrieve.py`  | Load / create a per-user FAISS vector store               |
| `faiss_update.py`    | Add, update, and delete records in a store                |
| `gen_label.py`       | Standalone structured-label extraction with Claude        |
| `process_new_note.py`| Experimental dedup using SentenceTransformers             |
| `faq_retrieve.py`    | Separate `/faq` endpoint for community Q&A over chat archives |
| `faq-vectorstore.py` | One-off script to build a FAISS index from a JSON export  |

Per-user data and embeddings are written under `static/faiss/` and
`static/cache/` (both git-ignored).

## Setup

### Prerequisites

- Python 3.10+
- [`ffmpeg`](https://ffmpeg.org/) (required by `pydub` for audio conversion)
- A Telegram bot token ([@BotFather](https://t.me/BotFather))
- OpenAI and Anthropic API keys

### Install

```bash
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root (it is git-ignored):

```dotenv
TELEGRAM_TOKEN=your-telegram-bot-token
OPENAI_KEY=your-openai-api-key
API_KEY=your-anthropic-api-key
WEBHOOK_URL=https://your-public-host/webhook
```

### Run

```bash
python app.py          # serves on http://localhost:5000
```

The Telegram Bot API requires a public HTTPS URL for webhooks. For local
development, expose port 5000 with a tunnel (e.g. `ngrok http 5000`), then
register the webhook:

```bash
curl -F "url=https://<your-public-url>/webhook" \
  "https://api.telegram.org/bot<TELEGRAM_TOKEN>/setWebhook"
```

## Security note

Never commit API keys. All keys are read from environment variables. If a key
is ever exposed in source or git history, **revoke and rotate it** — removing
it from the latest commit is not enough, as it remains in history.

## License

No license file is currently included. Add one before distributing.
