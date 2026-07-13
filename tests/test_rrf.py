from app.tools.rag_search import combine_rrf


class FakeChunk:
    """Stands in for a Chunk ORM object / SQLAlchemy Row — RRF only touches .id."""
    def __init__(self, id, chunk_text=""):
        self.id = id
        self.chunk_text = chunk_text

    def __repr__(self):
        return f"FakeChunk({self.id})"


def test_chunk_found_by_both_searches_ranks_first():
    ## the core claim of RRF: agreement between the two retrievers beats a
    ## strong showing in only one of them.
    vector = [FakeChunk(1), FakeChunk(2), FakeChunk(3)]
    keyword = [FakeChunk(3), FakeChunk(4), FakeChunk(5)]

    result = combine_rrf(vector, keyword, limit=5)

    # id=3 is the only chunk in both lists, so it should win
    assert result[0].id == 3


def test_rank_position_not_raw_score_decides():
    ## RRF deliberately ignores the underlying scores (cosine distance and ts_rank
    ## aren't on comparable scales) and fuses purely on rank position.
    vector = [FakeChunk(10), FakeChunk(20)]
    keyword = [FakeChunk(20), FakeChunk(10)]

    result = combine_rrf(vector, keyword, limit=2)

    # both appear once at rank 0 and once at rank 1 -> identical scores, both present
    assert {c.id for c in result} == {10, 20}


def test_limit_is_respected():
    vector = [FakeChunk(i) for i in range(10)]
    keyword = [FakeChunk(i) for i in range(10, 20)]

    assert len(combine_rrf(vector, keyword, limit=5)) == 5


def test_empty_keyword_results_falls_back_to_vector_order():
    vector = [FakeChunk(1), FakeChunk(2)]

    result = combine_rrf(vector, [], limit=5)

    assert [c.id for c in result] == [1, 2]


def test_both_empty_returns_empty():
    assert combine_rrf([], [], limit=5) == []
