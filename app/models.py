from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime 
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True)

    title = Column(String, nullable=False) ## Nullable = false because all articles need titles

    source_name = Column(String, nullable=False) ## will add filter/weight as to how reliable a source is

    published_date = Column(DateTime, nullable=False) ## help filter for recent news/freshness

    url = Column(String, unique=True, nullable=False) ## the natural key. NewsAPI returns the same article across overlapping queries, so the DB — not the caller — enforces "one row per article"


###--------------------------###

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"),nullable=False) ## article.id refers to the table above 'article'
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=False)
    sentiment_score = Column(Float, nullable=True)

    



    
