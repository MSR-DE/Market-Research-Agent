from google import genai
import os
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

logger = logging.getLogger(__name__)

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=2, max=120),   ## 2s → 4s → 8s → 16s … caps at 120s
    stop=stop_after_attempt(8),                             ## gives up after 8 tries (~4 min worst case)
    before_sleep=before_sleep_log(logger, logging.WARNING), ## logs each retry so you see what's happening
    reraise=True,
)
def embed_text(text):
    ## sending text to gemini's embedding model, and get back a vector representation
    result = client.models.embed_content(model="gemini-embedding-001", contents=text) ## 001 becuase its a text-only model, outputs 3072 dimension
    return result.embeddings[0].values ## take the first one [0], return .value (3072 floats)


if __name__ == "__main__":
    vec = embed_text("Apple stock rose sharply today.")
    print(f"Vector lenght: {len(vec)}") ## should be 3072
    print(f"First 5 values: {vec[:5]} ")
