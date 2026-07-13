from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from app.models import Base, Article, Chunk
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_text
import os
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv("DATABASE_URL")
engine = create_engine(database_url)
Session = sessionmaker(bind=engine)


def store_article(title, source_name, published_date, url, full_text):
    """Idempotent: storing the same URL twice is a no-op, and a mid-article failure
    leaves nothing behind. Safe to call again on a Celery retry."""
    session = Session()

    try:
        ## cheap guard for the common case: overlapping NewsAPI queries return the same
        ## article more than once. the unique constraint on url is the real backstop.
        if session.query(Article).filter_by(url=url).first():
            print(f"Skipped (already stored): '{title}'")
            return

        article = Article(
            title=title,
            source_name=source_name,
            published_date=published_date,
            url=url
        )
        session.add(article)
        session.flush()   ## assigns article.id WITHOUT committing — we still hold the transaction

        ## embed every chunk BEFORE committing anything. the old version committed the article
        ## first, so an embedding failure mid-article left an orphan article row behind — and
        ## the Celery retry then created a SECOND one. article + chunks now commit together
        ## or not at all.
        chunks = chunk_text(full_text)
        if not chunks:
            ## an article with no chunks is invisible to retrieval — it's a dead row that
            ## inflates the article count and tells the eval nothing. don't store it at all.
            print(f"Skipped (no chunkable text): '{title}'")
            session.rollback()
            return

        for chunk_str in chunks:
            vector = embed_text(chunk_str)
            session.add(Chunk(
                article_id=article.id,
                chunk_text=chunk_str,
                embedding=vector
            ))

        session.commit()
        print(f"Stored article '{title}' with {len(chunks)} chunks.")

    except IntegrityError:
        ## another worker won the race and inserted this URL between our check and our commit.
        ## that's the correct outcome — the article IS stored, just not by us.
        session.rollback()
        print(f"Skipped (concurrent insert): '{title}'")

    except Exception:
        session.rollback()   ## no half-written article survives a failure
        raise                ## let Celery see it and retry

    finally:
        session.close()      ## runs on every path, so the connection never leaks


if __name__ == "__main__":
    from datetime import datetime

    store_article(
        title="Test Article: Apple Stock News",
        source_name="Test Source",
        published_date=datetime.now(),
        url="https://example.com/test-article",
        full_text="Apple stock rose sharply today after strong earnings. " * 50
    )