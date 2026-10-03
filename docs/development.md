# Development and local setup

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | The code uses `X \| None` type hints; <!-- VERIFY: exact version used in development --> |
| Node.js (current LTS) | For the Next.js frontend |
| PostgreSQL with the `pgvector` extension | Embeddings are `vector(768)` |
| [Ollama](https://ollama.com) | Runs the chat model and the embedding model locally |
| Hardware | A machine that can run a 7B model at usable speed (GPU or a recent multi-core CPU with enough RAM) |

## 1. Models

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Ollama must be reachable at its default address (`127.0.0.1:11434`); one code path hard-codes this URL.

## 2. Database

```sql
CREATE DATABASE socia;
\c socia
CREATE EXTENSION IF NOT EXISTS vector;
```

<!-- VERIFY: how the schema is created (Alembic migration command or create_all). Document the exact command here. -->

## 3. Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

<!-- VERIFY: requirements file name/location and that .env.example exists. -->

Environment variables (from `app/core/config.py`):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy URL, e.g. `postgresql://user:pass@localhost:5432/socia` |
| `SECRET_KEY` | JWT signing key; use a long random value |
| `ALGORITHM` | JWT algorithm, e.g. `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token lifetime |
| `GEMINI_API_KEY` | Legacy: currently required at startup but unused. Slated for removal |

Check it is up: `GET http://localhost:8000/health`, and browse the interactive docs at `http://localhost:8000/docs`.

## 4. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

The frontend currently assumes the API at `http://localhost:8000` (hard-coded in login/register and the shared API helper), and the backend allows only the `http://localhost:3000` origin.

## Demo walkthrough

1. Register and complete the 5 onboarding questions.
2. Start a conversation and write a few messages about a recurring difficulty, for example dreading being judged in meetings on several occasions.
3. Wait a moment after each reply: extraction runs in the background.
4. Patterns appear once a tag has been seen 3 times **and** synthesis fires (every third behavior observation). To force it during a demo: `POST /api/v1/memory/synthesize` (with the bearer token).
5. Open **Progress**: view the emotion timeline and the pattern; choose **Add to Journey**.
6. Open the Journey; complete a challenge with a difficulty rating and a note, or skip with a reason, and observe that the next challenge responds.
7. Rate a few replies "not helpful" or a few challenges "too hard" and notice the softer chat tone once at least three signals exist.

## Tunable constants

| Constant | Value | File |
|---|---|---|
| Pattern promotion threshold | 3 | `services/memory_service.py` |
| Confidence-change threshold for a new version | 0.05 | `services/memory_service.py` |
| Retrieval limit / max cosine distance | 5 / 0.45 | `services/episodic_memory_service.py` |
| Embedding model | `nomic-embed-text` | `services/episodic_memory_service.py` |
| Chat / JSON model | `qwen2.5:7b` | `ai/ollama.py`, `services/*` |
| Recent-emotion window | 10 | `services/context_assembler.py` |
| Pacing minimum signals / window | 3 / 10 | `services/personalization_service.py` |
| Journey length bounds | 14-30 | `services/journey_service.py` |
| Extraction trigger | every 3rd behavior observation | `services/conversation_analyzer.py` |

Moving model names, URLs and these values into settings is a planned cleanup.

## Conventions

- Routers stay thin; logic lives in `services/`; the model adapter is `ai/ollama.py`.
- LLM output is never trusted: validate against a closed vocabulary before writing.
- Every new user-scoped query must filter by `user_id`.

## Testing

<!-- VERIFY: state what tests exist and how to run them. Recommended first tests (LLM mocked): safety regexes; pattern promotion rule and versioning; journey completion transitions; analyzer JSON validation; ownership checks. -->
