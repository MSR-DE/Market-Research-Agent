from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
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
    session = Session()

    
    article = Article(          ## saving the article first
        title=title,
        source_name=source_name,
        published_date=published_date,
        url=url
    )
    session.add(article)
    session.commit()   

    ## chunk the text, embed each piece, save as Chunk rows
    chunks = chunk_text(full_text)
    for chunk_str in chunks:
        vector = embed_text(chunk_str)
        chunk = Chunk(
            article_id=article.id,
            chunk_text=chunk_str,
            embedding=vector
        )
        session.add(chunk)

    session.commit()
    session.close()
    print(f"Stored article '{title}' with {len(chunks)} chunks.")


if __name__ == "__main__":
    from datetime import datetime

    store_article(
        title="Test Article: Apple Stock News",
        source_name="Test Source",
        published_date=datetime.now(),
        url="https://example.com/test-article",
        full_text="Apple stock rose sharply today after strong earnings. " * 50
    )