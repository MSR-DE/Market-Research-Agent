from google import genai
import os
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt
from google.genai.errors import ClientError

load_dotenv()

_client = None


def get_client():
    ## lazy init. constructing the client at module scope means this module can't be imported
    ## at all without a valid API key present, which breaks CI, unit tests, and anything that
    ## just wants to import a pure function from here. build it on first actual use instead.
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

def is_rate_limit(exception):
    # only retry on rate limits — a 404 or bad request will never succeed no matter how long we wait.
    # google-genai's ClientError exposes .status (a string like "RESOURCE_EXHAUSTED"), NOT a numeric code.
    if not isinstance(exception, ClientError):
        return False
    if getattr(exception, "status", None) == "RESOURCE_EXHAUSTED":
        return True
    return "429" in str(exception)   ## fallback: the message always carries the code


gemini_retry = retry(
    retry=retry_if_exception(is_rate_limit),
    wait=wait_exponential(multiplier=2, min=4, max=60),   # 4s → 8s → 16s → 32s → 60s
    stop=stop_after_attempt(5),
    reraise=True,
)

## same predicate, different patience. retry policy isn't one-size-fits-all — it depends on
## who is waiting. ingestion and eval are batch jobs with nobody watching, so they should back
## off hard and wait a rate limit out. but the agent serves a live UI request, and making a
## user stare at a spinner for ~60s before degrading is a WORSE outcome than degrading in 2s.
## a transient blip still gets one retry; a hard daily quota wall falls through fast.
gemini_retry_fast = retry(
    retry=retry_if_exception(is_rate_limit),
    wait=wait_exponential(multiplier=1, min=1, max=4),    # 1s, then give up
    stop=stop_after_attempt(2),
    reraise=True,
)

def _embed(text):
    ## sending text to gemini's embedding model, and get back a vector representation
    result = get_client().models.embed_content(model="gemini-embedding-001", contents=text) ## 001 because it's text-only, outputs 3072 dimensions
    return result.embeddings[0].values ## take the first one [0], return .values (3072 floats)


## same function, two retry policies, because the caller decides how patient to be.
embed_text = gemini_retry(_embed)         ## ingestion: batch, unattended — wait the rate limit out
embed_query = gemini_retry_fast(_embed)   ## live query: a user is waiting — degrade fast instead


if __name__ == "__main__":
    vec = embed_text("Apple stock rose sharply today.")
    print(f"Vector length: {len(vec)}") ## should be 3072
    print(f"First 5 values: {vec[:5]} ")
