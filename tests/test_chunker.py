from app.ingestion.chunker import chunk_text


def test_short_text_stays_one_chunk():
    assert chunk_text("hello world", chunk_size=300, overlap=50) == ["hello world"]


def test_chunks_respect_max_size():
    words = " ".join(str(i) for i in range(1000))
    chunks = chunk_text(words, chunk_size=300, overlap=50)
    assert all(len(c.split()) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    ## the whole point of overlap: a sentence split across a boundary should still
    ## appear intact in at least one chunk. so the tail of chunk N must reappear
    ## as the head of chunk N+1.
    words = " ".join(str(i) for i in range(700))
    chunks = chunk_text(words, chunk_size=300, overlap=50)

    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
