# SurfMind Backend

FastAPI service for SurfMind's retrieval pipeline. It accepts browser history
and bookmark payloads, caches them in Redis, and answers search requests using
hybrid retrieval with BM25, FAISS, and LLM-based post-processing.

## Stack

- FastAPI
- Redis
- LangChain
- FAISS
- OpenAI and Gemini

## Prerequisites

- Python 3.11
- Redis

## Setup

From the `backend/` directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Run

Start Redis, then launch the API:

```bash
uvicorn src.controller.main_controller:app --reload --host 0.0.0.0 --port 8000
```

Health checks:

- `GET /`
- `GET /health`

## API

Base router prefix: `/v1`

- `POST /v1/save-data`
  Saves history, bookmark, or combined payloads into Redis with a 1-hour TTL.
- `POST /v1/search`
  Runs the non-streaming RAG pipeline and returns a `SearchResponse`.
- `POST /v1/search-stream`
  Streams progress and final output as server-sent events.

## Request Shapes

`save-data` expects:

```json
{
  "userId": "user-123",
  "flag": "history",
  "data": [
    {
      "url": "https://example.com",
      "content": "Page content",
      "date": "2026-05-03",
      "domain": "example.com",
      "folder": "",
      "title": "Example"
    }
  ],
  "bookmarks": []
}
```

`search` and `search-stream` expect:

```json
{
  "userId": "user-123",
  "query": "what did I read about embeddings?",
  "flag": "history"
}
```

Supported `flag` values used by the backend are `history`, `bookmark`, and
`combined`.

## Notes

- Embeddings are provider-based and now support OpenAI/Gemini fallback.
- Redis is required for both save and search flows.
- Formatting and linting are configured through the repo root
  `.pre-commit-config.yaml` and `pyproject.toml`.
