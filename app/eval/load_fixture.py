import json
from app.ingestion.store import store_article

FIXTURE_PATH = "app/eval/fixtures/labeled_articles.json"


def load_fixture():
    # reads the committed JSON snapshot of the labeled eval articles and stores them
    # directly, bypassing NewsAPI, so a fresh clone gets the EXACT same data the
    # eval's test_set was labeled against (URLs are stable, unlike database ids).
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    for article in articles:
        store_article(
            article["title"],
            article["source_name"],
            article["published_date"],
            article["url"],
            article["full_text"],
        )
        print(f"Loaded fixture article: {article['title']}")


if __name__ == "__main__":
    load_fixture()
