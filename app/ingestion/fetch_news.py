import os
import requests
from app.ingestion.tasks import ingest_article_task
from dotenv import load_dotenv

load_dotenv() ## reads env file

api_key = os.getenv("NEWS_API_KEY")
url = "https://newsapi.org/v2/everything"


## deliberately overlapping queries around ONE topic cluster (rates / inflation / the Fed).
## a corpus of 100 articles about 100 different things is easy to retrieve from — the right
## answer is just the nearest vector by a mile. a corpus of 100 articles that all sound alike
## is what actually tests retrieval, and it's the only way the hybrid-vs-baseline ablation
## can say anything meaningful.
DEFAULT_QUERIES = [
    "Federal Reserve interest rates",
    "Fed rate cut decision",
    "FOMC meeting minutes",
    "Jerome Powell testimony",
    "inflation CPI data",
    "Treasury yields bonds",
    "CD rates savings APY",
    "mortgage rates housing market",
    "US economy recession outlook",
    "bank earnings net interest income",
]


def fetch_news(query, page_size=20):
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": page_size,   ## NewsAPI caps this at 100
        "language": "en",
        "sortBy": "publishedAt",
        "searchIn": "title,description"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()   ## a bad key or a blown quota should fail loudly, not silently KeyError
    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data.get('code')} — {data.get('message')}")

    return data.get("articles", [])


def run_ingestion(queries=None, page_size=20):
    queries = queries or DEFAULT_QUERIES

    ## dedupe across queries BEFORE queueing. these searches overlap heavily by design, so the
    ## same article comes back under several of them. store_article is idempotent and would
    ## skip the repeats anyway, but there's no reason to burn Celery tasks and embedding
    ## quota discovering that.
    seen_urls = set()
    queued = 0
    skipped = 0

    for query in queries:
        try:
            articles = fetch_news(query, page_size=page_size)
        except Exception as e:
            print(f"[Fetch] Query '{query}' failed ({e}), continuing")
            continue

        for article in articles:
            article_url = article.get("url")
            if not article_url or article_url in seen_urls:
                skipped += 1
                continue

            body = (article.get("description") or "") + " " + (article.get("content") or "")
            if not body.strip():
                skipped += 1      ## nothing to embed
                continue

            seen_urls.add(article_url)

            ingest_article_task.delay(
                article["title"],
                article["source"]["name"],
                article["publishedAt"],
                article_url,
                body
            )
            queued += 1

        print(f"[Fetch] '{query}' -> {len(articles)} results (running total queued: {queued})")

    print(f"\nQueued {queued} unique articles. Skipped {skipped} duplicates/empties.")
    return queued


if __name__ == "__main__":
    run_ingestion()
