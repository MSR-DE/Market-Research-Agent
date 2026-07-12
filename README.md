![Python](https://img.shields.io/badge/Python-3.13-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Celery](https://img.shields.io/badge/Celery-Redis-green)
![Gemini](https://img.shields.io/badge/Google-Gemini_API-orange)

# Market Research Agent

An agentic RAG system for financial news research. Given a question, the agent
decides — using real function-calling, not a fixed script — whether it needs to
search a corpus of ingested news articles, retrieves relevant information using
hybrid search (vector + keyword) with LLM reranking, and answers grounded in real
retrieved data. Includes an eval harness measuring retrieval accuracy and answer
quality against a hand-labeled test set.

---

## Architecture Overview

```mermaid
graph TD
    User([User Query]) --> Agent[Agent Orchestrator]
    Agent -->|Decides to search| Tool[search_news Tool]
    Agent -->|Has enough info| Answer([Final Answer])

    Tool --> Hybrid[Hybrid Search]
    Hybrid --> Vector[Vector Search - cosine distance]
    Hybrid --> Keyword[Postgres Full-Text Search]
    Vector --> RRF[Reciprocal Rank Fusion]
    Keyword --> RRF
    RRF --> Rerank[LLM Reranking]
    Rerank --> Tool

    Tool --> DB[(PostgreSQL + pgvector)]

    NewsAPI[NewsAPI] --> Fetch[Fetch]
    Fetch --> Celery[Celery Worker]
    Celery -->|Chunk| Chunker[Chunker]
    Chunker -->|Embed| Embedder[Gemini Embeddings]
    Embedder -->|Store| DB
    Redis[(Redis Broker)] --> Celery
```

---

## Tech Stack & Concepts Demonstrated

* **PostgreSQL & pgvector**: vector storage with cosine-distance similarity search, combined with native full-text search for hybrid retrieval.
* **Reciprocal Rank Fusion (RRF)**: combines vector and keyword search results by rank position rather than raw score, since the two scoring scales aren't directly comparable.
* **Celery & Redis**: background ingestion, one task per article (not per pipeline stage — articles are independent, stages are sequential), with retry/backoff on failure.
* **Google Gemini (GenAI SDK)**: `gemini-embedding-001` for 3072-dim embeddings, `gemini-2.5-flash` for reranking, agent reasoning, and answer generation via function-calling.
* **Agentic tool-use**: the agent decides per-query whether to call the search tool, using Gemini's native function-calling API — not a hardcoded retrieval step.
* **Eval harness**: hand-labeled retrieval accuracy test, LLM-as-judge answer quality scoring (checked against actual retrieved source material, not judged in isolation), and a baseline ablation comparing hybrid+RRF+rerank against naive vector search.
* **Docker & Docker Compose**: orchestrates PostgreSQL (with pgvector) and Redis.

---

## Project Structure

```
├── app/
│   ├── models.py              # SQLAlchemy models: Article, Chunk (with Vector column)
│   ├── celery_app.py          # Celery configuration (Redis broker + backend)
│   ├── init_db.py             # Creates tables from models
│   ├── ingestion/
│   │   ├── fetch_news.py      # NewsAPI fetch + run_ingestion orchestration
│   │   ├── chunker.py         # Text chunking with overlap
│   │   ├── embedder.py        # Gemini embedding calls
│   │   ├── store.py           # Article + Chunk persistence
│   │   └── tasks.py           # Celery task: one per article, retry w/ backoff
│   ├── tools/
│   │   └── rag_search.py      # Vector search, keyword search, RRF, reranking
│   ├── agent/
│   │   └── orchestrator.py    # Agent loop with Gemini function-calling
│   └── eval/
│       └── eval_runner.py     # Retrieval accuracy, answer quality, baseline ablation
├── docker-compose.yml          # PostgreSQL (pgvector) + Redis
├── requirements.txt
└── .env                        # NEWS_API_KEY, GEMINI_API_KEY, DATABASE_URL (not committed)
```

---

## Getting Started

### Prerequisites
* Docker & Docker Compose
* A Google Gemini API Key
* A NewsAPI key ([newsapi.org](https://newsapi.org))

### 1. Configure the Environment
Create a `.env` file:
```env
NEWS_API_KEY=your-newsapi-key
GEMINI_API_KEY=your-gemini-api-key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/market_research
REDIS_URL=redis://localhost:6379/0
```

### 2. Start Postgres + Redis
```bash
docker compose up -d
```

### 3. Enable pgvector and create tables
```bash
docker exec -it market-research-db psql -U postgres -d market_research -c "CREATE EXTENSION IF NOT EXISTS vector;"
python -m app.init_db
```

### 4. Ingest articles
```bash
# in one terminal — the Celery worker
celery -A app.celery_app worker --loglevel=info --pool=solo

# in another terminal — fetch and queue articles
python -m app.ingestion.fetch_news
```

### 5. Ask the agent something
```bash
python -m app.agent.orchestrator
```

### 6. Run the eval harness
```bash
python -m app.eval.eval_runner
```

---

## Results

* **Retrieval accuracy**: 100% on a hand-labeled test set
* **Answer quality**: 5.0/5 average (LLM-judged against actual retrieved source material, not judged in isolation)
* **Baseline ablation**: hybrid+RRF+rerank performed equal to naive vector search on the current test set — see Limitations

---

## Known Limitations

* **Single-turn only** — no conversation memory across queries yet; each question is independent.
* **Small eval test set** (6 questions) — sufficient to validate the harness works, not yet large or hard enough to reliably isolate hybrid search's advantage over naive vector search. Harder cases (paraphrased queries, near-duplicate articles) are the natural next step.
* **No deduplication** — two ingested articles are exact duplicates from overlapping NewsAPI search queries.
* **Free-tier Gemini rate limits** materially slow batch eval runs (not an issue for normal single-query use, which makes 2-4 calls).
* **Judge consistency not yet verified** — the LLM-as-judge hasn't been checked for score stability across repeated runs of the same case.

---

## Code Architecture Walkthrough

* **Agent loop**: [app/agent/orchestrator.py](app/agent/orchestrator.py) — sends the full conversation history to Gemini each turn; a `function_call` in the response triggers a tool call, a plain-text response is the stop condition. A `max_iterations` cap is the safety net if the model never naturally stops.
* **Hybrid retrieval**: [app/tools/rag_search.py](app/tools/rag_search.py) — combines
  ```python
  Chunk.embedding.cosine_distance(query_vector)
  ```
  (vector search) with Postgres `ts_rank`/`plainto_tsquery` (keyword search) via Reciprocal Rank Fusion, then reranks the merged list with an LLM call.
* **Ingestion as background tasks**: [app/ingestion/tasks.py](app/ingestion/tasks.py) — one Celery task per article (fetch → chunk → embed → store), with retry and exponential backoff on failure.
