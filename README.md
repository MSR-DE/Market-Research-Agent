![Python](https://img.shields.io/badge/Python-3.13-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Celery](https://img.shields.io/badge/Celery-Redis-green)
![Gemini](https://img.shields.io/badge/Google-Gemini_API-orange)

# Market Research Agent

An agentic RAG system for financial news research. Given a question, the agent decides for itself — through real function-calling, not a scripted retrieval step — whether it needs to search a corpus of ingested news articles. Retrieval is hybrid (vector + keyword) fused with Reciprocal Rank Fusion and reranked by an LLM, and answers are grounded in what was actually retrieved.

The part I care most about: it ships with an eval harness. Retrieval accuracy against a hand-labeled test set, an ablation against a naive vector-search baseline, no-answer detection, and LLM-as-judge answer scoring. Most RAG demos skip this entirely.
---

## Architecture

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

## What's in here, and why it's built that way

**Agentic tool use.** The agent uses Gemini's native function-calling API and decides per query whether to call the search tool. Ask it something answerable from general knowledge and it just answers; ask about a specific news event and it searches. The loop feeds tool results back into the conversation and lets the model take another turn, with a `max_iterations` cap as the safety net.

**Hybrid retrieval.** Vector search (pgvector, cosine distance over 3072-dim `gemini-embedding-001` embeddings) catches paraphrases; Postgres full-text search catches exact terms like "4.10% APY" that embeddings are fuzzy about. The two result lists are merged with Reciprocal Rank Fusion — by rank position, not raw score, because cosine distance and `ts_rank` aren't on comparable scales. The fused list then gets reranked by an LLM call.

**Rate limits handled properly, not with sleeps.** Every Gemini call is wrapped in a tenacity retry policy that fires only on 429 / `RESOURCE_EXHAUSTED`, with exponential backoff (4s to 60s, five attempts). A 404 or a bad request fails immediately — retrying those would just waste time. An earlier version of this code used hardcoded `time.sleep(25)` before every call; the retry policy replaced it.

**Graceful degradation, verified for real.** If Gemini is unavailable after retries are exhausted, the agent falls back to raw vector search results instead of crashing. This isn't theoretical — it was exercised live during an actual quota exhaustion. Same philosophy one layer down: if the LLM reranker returns output that can't be parsed, retrieval falls back to the RRF ordering, which is already a reasonable ranking.

**Ingestion as background work.** Fetching from NewsAPI queues one Celery task per article — per article, not per pipeline stage, because articles are independent of each other while the stages within one article (chunk → embed → store) are sequential. Tasks retry with exponential backoff on failure. Redis is the broker.

**An eval harness that can say "no".** Beyond retrieval accuracy and answer quality, there's a no-answer test: questions with no correct answer in the corpus, scored by cosine-distance threshold rather than ID matching, because ID matching can't express "nothing here is relevant." There's also a judge-consistency check that scores the same case multiple times to see whether the LLM judge is stable. If the judge returns something unparseable, that case is excluded from the average rather than silently miscounted.

---

## Project structure

```
├── app/
│   ├── models.py              # SQLAlchemy models: Article, Chunk (with Vector column)
│   ├── celery_app.py          # Celery configuration (Redis broker + backend)
│   ├── init_db.py             # Creates tables from models
│   ├── ingestion/
│   │   ├── fetch_news.py      # NewsAPI fetch + ingestion orchestration
│   │   ├── chunker.py         # Text chunking with overlap
│   │   ├── embedder.py        # Gemini embeddings + the shared 429-only retry policy
│   │   ├── store.py           # Article + Chunk persistence
│   │   └── tasks.py           # Celery task: one per article, retry w/ backoff
│   ├── tools/
│   │   └── rag_search.py      # Vector search, keyword search, RRF, LLM reranking
│   ├── agent/
│   │   └── orchestrator.py    # Agent loop with Gemini function-calling
│   ├── eval/
│   │   └── eval_runner.py     # Retrieval accuracy, baseline ablation, no-answer detection, LLM-as-judge
│   └── ui.py                  # Minimal Streamlit front end with the agent's reasoning trace
├── docker-compose.yml          # PostgreSQL (pgvector) + Redis
├── requirements.txt
└── .env                        # NEWS_API_KEY, GEMINI_API_KEY, DATABASE_URL (not committed)
```

---

## Getting started

### Prerequisites
* Docker & Docker Compose
* A Google Gemini API key
* A NewsAPI key ([newsapi.org](https://newsapi.org))

### 1. Configure the environment
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
# terminal 1 — the Celery worker
celery -A app.celery_app worker --loglevel=info --pool=solo

# terminal 2 — fetch and queue articles
python -m app.ingestion.fetch_news
```

### 5. Ask the agent something
```bash
python -m app.agent.orchestrator
```
Or with the UI:
```bash
streamlit run app/ui.py
```

### 6. Run the eval harness
```bash
python -m app.eval.eval_runner
```

---

## Results

The test set is 11 hand-labeled questions plus a separate no-answer set. It deliberately includes cases designed to separate hybrid search from the vector-only baseline: a paraphrased query with none of the source article's key terms, an exact-number query ("4.10% APY") that should favor keyword search, and a duplicate-article case where either copy counts as a hit.

<!-- last run: today, on the current 11-question set -->
* **Retrieval accuracy (hybrid + RRF + rerank)**: 11/11 (100%)
* **Baseline (vector-only) retrieval accuracy**: 11/11 (100%) — tied with hybrid on hit-rate, though hybrid tightened ranking position on several ambiguous cases (one query went from a 3-candidate spread down to a single confident match). At this corpus size, vector search alone is already strong enough to place the right article in the top 3; a larger, noisier corpus would be a better test of hybrid's actual advantage.
* **No-answer detection**: 0/1 — a query entirely outside the corpus ("What is Tesla's current stock price?") returned a closest cosine distance of 0.342 against a 0.7 threshold, meaning pure vector search still surfaced confident-looking results for a topic the corpus has nothing on.
* **Answer quality**: partial (2/5 and 5/5 on the first two cases; free-tier rate limits interrupted the run before the remaining 9). Full run pending billing setup — see Known limitations.

---

## Known limitations

* **Single-turn only.** No conversation memory across queries yet; each question is independent.
* **The test set is still small** (11 questions + 1 no-answer case). Big enough to catch regressions and exercise the paraphrase/keyword/duplicate edge cases, not big enough for the accuracy numbers to be statistically strong. Growing it is ongoing.
* **Eval article IDs are instance-specific.** The expected-answer labels reference article IDs from my local ingestion run, so the eval isn't reproducible from a fresh clone yet. A committed fixture of the labeled articles is the fix, and it's next on the list.
* **No deduplication on ingest.** Overlapping NewsAPI queries can store the same article twice. A unique constraint on URL would fix most of it.
* **Free-tier Gemini rate limits** make batch eval runs slow even with backoff. Normal single-query use is unaffected (2–4 API calls).
* **Judge consistency is checkable but not yet systematically verified.** The harness includes a repeated-scoring consistency check; I haven't yet run it across the full test set and published the variance.
* **The fallback path is relevance-blind.** When Gemini is down, raw vector search always returns the *closest* content in the corpus, even when nothing in the corpus is actually relevant. The no-answer eval exists precisely because of this failure mode.

---

## Code walkthrough

* **Agent loop** — [app/agent/orchestrator.py](app/agent/orchestrator.py). Full conversation history goes to Gemini each turn. A `function_call` in the response triggers a tool call; a plain-text response is the stop condition; `max_iterations` is the safety net.
* **Hybrid retrieval** — [app/tools/rag_search.py](app/tools/rag_search.py). Combines `Chunk.embedding.cosine_distance(query_vector)` with Postgres `ts_rank`/`plainto_tsquery` via RRF, then reranks with an LLM call that falls back to RRF order on unparseable output.
* **Retry policy** — [app/ingestion/embedder.py](app/ingestion/embedder.py). One shared tenacity decorator, scoped to rate-limit errors only, used by the embedder, the reranker, the agent, and the judge.
* **Ingestion tasks** — [app/ingestion/tasks.py](app/ingestion/tasks.py). One Celery task per article with retry and exponential backoff.
