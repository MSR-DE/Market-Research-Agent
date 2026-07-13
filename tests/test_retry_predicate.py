from app.ingestion.embedder import is_rate_limit


class FakeClientError(Exception):
    """Mimics google.genai.errors.ClientError, which carries .status as a STRING."""
    def __init__(self, status=None, message=""):
        super().__init__(message)
        self.status = status


def _patched_is_rate_limit(exc, monkeypatch):
    import app.ingestion.embedder as embedder
    monkeypatch.setattr(embedder, "ClientError", FakeClientError)
    return embedder.is_rate_limit(exc)


def test_resource_exhausted_is_retried(monkeypatch):
    exc = FakeClientError(status="RESOURCE_EXHAUSTED", message="quota exceeded")
    assert _patched_is_rate_limit(exc, monkeypatch) is True


def test_429_in_message_is_retried(monkeypatch):
    exc = FakeClientError(status=None, message="429 Too Many Requests")
    assert _patched_is_rate_limit(exc, monkeypatch) is True


def test_404_is_not_retried(monkeypatch):
    ## the whole reason this predicate exists. a 404 will NEVER succeed no matter
    ## how long we back off, so retrying it just burns 60+ seconds for nothing.
    exc = FakeClientError(status="NOT_FOUND", message="404 model not found")
    assert _patched_is_rate_limit(exc, monkeypatch) is False


def test_unrelated_exception_is_not_retried():
    assert is_rate_limit(ValueError("something else went wrong")) is False
