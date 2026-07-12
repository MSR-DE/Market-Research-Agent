import os 
import requests
from app.ingestion.tasks import ingest_article_task
from dotenv import load_dotenv

load_dotenv() ## reads env file

api_key = os.getenv("NEWS_API_KEY")
url = "https://newsapi.org/v2/everything" 

def fetch_news(query):
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": 5,
        "language": "en",
        "sortBy": "publishedAt",
        "searchIn": "title,description"
    }

    response = requests.get(url, params=params)
    data = response.json()

    return data["articles"]   # NEW: return the raw article list instead of printing 



def run_ingestion(queries):
    for query in queries:
        articles = fetch_news(query)
        for article in articles:
            ingest_article_task.delay(
                article["title"],
                article["source"]["name"],
                article["publishedAt"],
                article["url"],
                (article.get("description") or "") + " " + (article.get("content") or "")
            )
            print(f"Queued: {article['title']}")


if __name__ == "__main__":
    run_ingestion(["Apple", "Federal Reserve", "stock market"])

