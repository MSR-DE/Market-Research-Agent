from app.ingestion.store import Session
from app.models import Chunk
from app.ingestion.embedder import embed_query, get_client, gemini_retry_fast
from sqlalchemy import text


## cosine search
def search_chunk(query, limit=5):

    query_vector = embed_query(query) ## turning the user's query into a vector — fast-degrade policy, a user is waiting

    session = Session()

    results = (                     ## getting postgres for the chunks that the embedding are closest in meaning
        session.query(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(limit).all()                                           ## uses cosine_distance = angle b/w vectors, not raw magnitutate or distance
    )

    session.close()

    return results




###-----------------------------###    


## simple keyword search 

def keyword_search(query, limit=5):

    session = Session()

    sql = text(             ## converts the text into a searchable format (stems words, removes common filler words like "the"/"a").
        """
        SELECT id, article_id, chunk_text,
        ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', :query)) AS rank
        FROM chunks
        WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :limit
""")

    results = session.execute(sql, {"query": query, "limit": limit}).fetchall()
    session.close()
    return results



###---------------------------###


def combine_rrf(vector_results, keyword_results, k=60, limit=5): ## using RRF to sort the best one accordingly
    scores = {}
    chunks_by_id = {}

    for rank, chunk in enumerate(vector_results): ## rank 0 = best match in each list
        chunks_by_id[chunk.id] = chunk
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (rank + k)

        
    for rank, row in enumerate(keyword_results):
        chunks_by_id.setdefault(row.id, row)
        scores[row.id] = scores.get(row.id, 0) + 1 / (rank + k)

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    return [chunks_by_id[cid] for cid in sorted_ids[:limit]]

def hybrid_search(query, limit=5):
    vector_results = search_chunk(query, limit=10)
    keyword_results = keyword_search(query, limit=10)
    return combine_rrf(vector_results, keyword_results, limit=limit)


###-----------------------### 
@gemini_retry_fast   ## also on the interactive path — and rerank already degrades safely to RRF order
def _call_reranker(prompt):
    return get_client().models.generate_content(model="gemini-2.5-flash", contents=prompt)


def rerank(query, chunks, top_n=5):
    if not chunks:
        return []

    ## build numbered list of chunk texts for the LLM to judge
    chunk_list = "\n".join(f"[{i}] {chunk.chunk_text[:300]}" for i, chunk in enumerate(chunks))

    prompt = f"""Query: {query}

    Below are numbered text chunks. Rank them by how relevant they are to answering the query.
    Return ONLY a comma-separated list of chunk numbers, most relevant first. No explanation.

    {chunk_list}"""

    ## two ways this can fail, and BOTH must degrade to RRF order rather than crash:
    ##   1. the API call itself dies (429 quota, network, timeout)
    ##   2. the call succeeds but returns unparseable text — "Here are the rankings: 2, 0, 1"
    ##      instead of "2,0,1", or garbage entirely
    ## reranking is an *optimisation* over RRF, not a requirement. if it's unavailable, the
    ## RRF ordering is already a reasonable ranking, so we return that and keep serving.
    try:
        response = _call_reranker(prompt)
        order = [int(i.strip()) for i in response.text.strip().split(",")]
        reranked = [chunks[i] for i in order if i < len(chunks)]
        if not reranked:                      ## parsed, but nothing valid came out
            raise ValueError("no valid indices in reranker output")
    except Exception as e:
        print(f"[Rerank] Unavailable ({type(e).__name__}: {e}), falling back to RRF order")
        reranked = chunks                     ## RRF order is already a decent ranking

    return reranked[:top_n]

def hybrid_search_reranked(query, limit=5):
    candidates = hybrid_search(query, limit=10)   # get more candidates than needed
    return rerank(query, candidates, top_n=limit)


if __name__ == "__main__":
    matches = hybrid_search_reranked("Fed Reserve interest rate decision?")
    for chunk in matches:
        print(f"[article_id={chunk.article_id}] {chunk.chunk_text[:100]}")
