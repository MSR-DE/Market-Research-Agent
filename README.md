![CI](https://github.com/msr-de/Market-Research-Agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
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

**Idempotent ingestion.** `url` is a unique key, and an article commits in the *same transaction* as its chunks — so a mid-article embedding failure rolls back cleanly and a Celery retry can't leave a duplicate or an orphaned article with no chunks. Verified: ingesting 109 articles across ten heavily-overlapping NewsAPI queries produced **zero duplicate URLs and zero orphaned articles**. (The earlier version committed the article first and then embedded, which did exactly the wrong thing under retry.)

**Ingestion as background work.** Fetching from NewsAPI queues one Celery task per article — per article, not per pipeline stage, because articles are independent of each other while the stages within one article (chunk → embed → store) are sequential. Tasks retry with exponential backoff on failure. Redis is the broker.

**Tests that don't need the LLM.** You can't unit-test a language model's output, but you *can* test everything wrapped around it — and that's where the bugs actually live. So CI covers the deterministic parts: chunk overlap boundaries, RRF fusion order, the reranker's fallback when the model returns prose instead of `"2,0,1"`, and the retry predicate correctly telling a 429 apart from a 404. No database, no API calls, no flakiness.

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
│   │   ├── eval_runner.py     # Retrieval accuracy, baseline ablation, no-answer detection, LLM-as-judge
│   │   ├── load_fixture.py    # Loads the labeled corpus so the eval reproduces from a fresh clone
│   │   └── fixtures/labeled_articles.json
│   └── ui.py                  # Minimal Streamlit front end with the agent's reasoning trace
├── tests/                      # pure-logic unit tests — no DB, no API key needed
│   ├── test_chunker.py         # chunk sizing + overlap invariants
│   ├── test_rrf.py             # rank fusion: agreement between retrievers wins
│   ├── test_rerank_fallback.py # LLM returns garbage -> degrade to RRF order, don't crash
│   └── test_retry_predicate.py # retry on 429, never on 404
├── .github/workflows/ci.yml    # install -> pytest -> docker build
├── Dockerfile
├── docker-compose.yml          # PostgreSQL (pgvector) + Redis + app + worker
├── requirements.txt
├── .dockerignore               # keeps .env and the local venvs out of the image layers
├── .env.example                # copy to .env and fill in
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

### Or just run the whole stack
```bash
docker compose up --build
```
Brings up Postgres, Redis, the Streamlit app, and the Celery worker together. The app and worker wait on healthchecks, so the worker doesn't race Postgres to the connection.

### Run the tests
```bash
pytest
```
No database and no API key required — the Gemini client is lazily constructed, and the LLM calls are stubbed.

---

## Results

Corpus: **109 articles** ingested from NewsAPI across ten deliberately overlapping queries around one topic cluster (Fed / rates / inflation / housing), plus a committed fixture of the hand-labeled gold articles. The overlap is the point — a corpus of 109 articles about 109 different things is trivial to retrieve from, because the right answer is the nearest vector by a mile. A corpus where a hundred articles all discuss rate cuts is what actually tests retrieval.

Test set: 11 hand-labeled questions plus a no-answer case. Labels reference article **URLs**, not database IDs, and the labeled articles are committed as a fixture — so these numbers reproduce from a fresh clone:

```bash
python -m app.eval.load_fixture     # gold articles first
python -m app.ingestion.fetch_news  # then the distractors
python -m app.eval.eval_runner
```

| Arm | Retrieval accuracy |
|---|---|
| Vector-only (baseline) | **82%** (9/11) |
| Hybrid + RRF | **82%** (9/11) |
| Hybrid + RRF + LLM rerank | *pending — free-tier quota* |

**On an earlier ~20-article corpus, both arms scored 100%.** That number was worthless: the corpus was too small to fail at. Enlarging it to 109 semantically crowded articles is what made the eval able to say anything at all.

**The two failures are the interesting part, and they aren't retrieval failures.** Both arms missed the same two questions:

* *"What did a Federal Reserve official say about cutting interest rates?"* → returned three articles about Fed officials discussing rate cuts. Just not the one labeled correct.
* *"What was the CD account interest rate mentioned recently?"* → returned three articles about CD rates. Just not the one labeled correct.

Ingesting 109 articles on one topic means a dozen articles now legitimately answer "what did a Fed official say about rate cuts?" — but the test set asserts exactly one gold URL per question. Retrieval worked; **the labeling assumption broke.** The control that proves this: `"4.10% APY"` still passes, because that question has a genuinely unique answer and survives the crowded corpus. The two that broke are precisely the two whose answers stopped being unique.

The fix isn't better retrieval, it's a better metric — multi-label gold sets, or graded relevance (nDCG) instead of exact-match hit rate. That's the next thing on the list.

**No-answer detection: 0/1.** A query with no answer in the corpus ("What is Tesla's current stock price?") returned a closest cosine distance of 0.378 against a 0.7 threshold. Vector search has no way to say "nothing here is relevant" — it always returns the nearest chunk, and 0.378 looks confident. See limitations.

**Answer quality (LLM-as-judge): pending.** Free-tier quota (20 `generate_content` requests/day) is exhausted by the retrieval arms before the judged run completes.

---

## Known limitations

* **Single-turn only.** No conversation memory across queries yet; each question is independent.
* **The eval metric is too strict for the corpus.** Exact-match hit rate against a single gold URL per question stops being meaningful once a hundred articles discuss the same topic — see Results. Multi-label gold sets or graded relevance (nDCG) is the fix.
* **The test set is still small** (11 questions + 1 no-answer case). Enough to catch regressions and expose the labeling problem above, not enough for the accuracy numbers to be statistically strong.
* **The no-answer case fails, and it's the most useful result in here.** Vector search has no concept of "nothing here is relevant" — it always returns the *closest* chunk, and a 0.378 distance looks confident even for a query the corpus can't answer. A distance threshold alone isn't sufficient; the real fix is a relevance gate before generation. Highest-priority item.
* **The corpus is shallow.** NewsAPI's free tier truncates article bodies to roughly 200 characters, so each stored article is effectively a headline plus a description (~440 chars, one chunk). The chunking-with-overlap logic is correct and unit-tested, but this corpus never exercises it. Full-text scraping or a paid tier would give a corpus where multi-chunk retrieval actually matters.
* **Answer quality is unmeasured.** Free-tier quota (20 `generate_content` requests/day) runs out before the LLM-as-judge pass completes. Needs a paid key.
* **Retry patience is split by caller, deliberately.** Ingestion and eval are unattended batch jobs, so they back off hard (4s→60s, 5 attempts) and wait a rate limit out. The agent serves a live UI request, so it degrades in ~2s instead — making a user watch a spinner for a minute before falling back is a worse outcome than falling back immediately. Same retry predicate, different patience.
* **Free-tier Gemini rate limits** make batch eval runs slow even with backoff. Normal single-query use is unaffected (2–4 API calls).
* **Judge consistency is checkable but not yet systematically verified.** The harness includes a repeated-scoring consistency check; I haven't yet run it across the full test set and published the variance.
* **The fallback path is relevance-blind.** When Gemini is down, raw vector search always returns the *closest* content in the corpus, even when nothing in the corpus is actually relevant. The no-answer eval exists precisely because of this failure mode.

---

## Code walkthrough

* **Agent loop** — [app/agent/orchestrator.py](app/agent/orchestrator.py). Full conversation history goes to Gemini each turn. A `function_call` in the response triggers a tool call; a plain-text response is the stop condition; `max_iterations` is the safety net.
* **Hybrid retrieval** — [app/tools/rag_search.py](app/tools/rag_search.py). Combines `Chunk.embedding.cosine_distance(query_vector)` with Postgres `ts_rank`/`plainto_tsquery` via RRF, then reranks with an LLM call that falls back to RRF order on unparseable output.
* **Retry policy** — [app/ingestion/embedder.py](app/ingestion/embedder.py). One shared tenacity decorator, scoped to rate-limit errors only, used by the embedder, the reranker, the agent, and the judge.
* **Ingestion tasks** — [app/ingestion/tasks.py](app/ingestion/tasks.py). One Celery task per article with retry and exponential backoff.
