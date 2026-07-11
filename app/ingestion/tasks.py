from app.celery_app import celery_app
from app.ingestion.store import store_article

@celery_app.task(name="ingest_article", bind=True, max_retries=3)
def ingest_article_task(self, title, source_name, published_date, url, full_text):
    try:
        store_article(title, source_name, published_date, url, full_text)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

