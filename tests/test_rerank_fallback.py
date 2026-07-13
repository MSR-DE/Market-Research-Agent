import app.tools.rag_search as rag_search


class FakeChunk:
    def __init__(self, id, chunk_text=""):
        self.id = id
        self.chunk_text = chunk_text


class FakeResponse:
    def __init__(self, text):
        self.text = text


CHUNKS = [FakeChunk(0, "alpha"), FakeChunk(1, "beta"), FakeChunk(2, "gamma")]


def test_clean_output_is_used_to_reorder(monkeypatch):
    monkeypatch.setattr(rag_search, "_call_reranker", lambda p: FakeResponse("2,0,1"))

    result = rag_search.rerank("q", CHUNKS, top_n=3)

    assert [c.id for c in result] == [2, 0, 1]


def test_chatty_output_falls_back_to_rrf_order(monkeypatch):
    ## the failure this guards against: the model ignores "return ONLY numbers"
    ## and prefixes an explanation. int() blows up, and we must NOT crash — RRF
    ## order is already a reasonable ranking, so we degrade to it.
    monkeypatch.setattr(
        rag_search, "_call_reranker", lambda p: FakeResponse("Here are the rankings: 2, 0, 1")
    )

    result = rag_search.rerank("q", CHUNKS, top_n=3)

    assert [c.id for c in result] == [0, 1, 2]   # unchanged RRF order


def test_garbage_output_falls_back_to_rrf_order(monkeypatch):
    monkeypatch.setattr(
        rag_search, "_call_reranker", lambda p: FakeResponse("I cannot rank these.")
    )

    result = rag_search.rerank("q", CHUNKS, top_n=3)

    assert [c.id for c in result] == [0, 1, 2]


def test_out_of_range_indices_are_dropped(monkeypatch):
    ## the model hallucinates a chunk number that doesn't exist
    monkeypatch.setattr(rag_search, "_call_reranker", lambda p: FakeResponse("1,99,0"))

    result = rag_search.rerank("q", CHUNKS, top_n=3)

    assert [c.id for c in result] == [1, 0]


def test_api_failure_falls_back_to_rrf_order(monkeypatch):
    ## the reranker API itself dies (429 quota exhausted, network, timeout).
    ## reranking is an optimisation over RRF, not a requirement — so retrieval must
    ## keep serving on the RRF ordering instead of taking the whole request down.
    def blow_up(prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(rag_search, "_call_reranker", blow_up)

    result = rag_search.rerank("q", CHUNKS, top_n=3)

    assert [c.id for c in result] == [0, 1, 2]   # unchanged RRF order, no exception


def test_empty_chunks_short_circuits():
    ## must not call the LLM at all when there's nothing to rank
    assert rag_search.rerank("q", [], top_n=3) == []
