from app.ingestion.store import Session
from app.models import Chunk
from app.ingestion.embedder import embed_text



def search_chunk(query, limit=5):

    query_vector = embed_text(query) ## turning users query into a vector

    session = Session()

    results = (                     ## getting postgres for the chunks that the embedding are closest in meaning
        session.query(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(limit).all()                                           ## uses cosine_distance = angle b/w vectors, not raw magnitutate or distance
    )

    session.close()

    return results


if __name__ == "__main__":
    matches = search_chunk("Fed Reserve interest rate decision?")
    for chunk in matches:
        print(f"[article_id={chunk.article_id}] {chunk.chunk_text[:100]}")
