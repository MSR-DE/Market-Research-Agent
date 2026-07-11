from celery import Celery
import os 
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0") ## easy fallback incase env is empty

celery_app = Celery(
    "market_research",  ## just a name for this celery app
    broker=redis_url,  ## where tasks will get queued 
    backend=redis_url,  ## where task Results get stored
    include=["app.ingestion.tasks"],   ## import this module and register its tasks
)


celery_app.conf.update(
    task_serializer="json",  ## how tasts args get encoded in redis
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True, ## to see the running state
)


celery_app.autodiscover_tasks(["app.ingestion"])
