from google import genai
import os
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt
from google.genai.errors import ClientError

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

def is_rate_limit(exception):
    # only retry on rate limits — a 404 or bad request will never succeed no matter how long we wait.
    # google-genai's ClientError exposes .status (a string like "RESOURCE_EXHAUSTED"), not a numeric code.
    if not isinstance(exception, ClientError):
        return False
    if getattr(exception, "status", None) == "RESOURCE_EXHAUSTED":
        return True
    return "429" in str(exception)   ## belt-and-braces: the message always carries the code


gemini_retry = retry(
    retry=retry_if_exception(is_rate_limit),
    wait=wait_exponential(multiplier=2, min=4, max=60),   # 4s → 8s → 16s → 32s → 60s
    stop=stop_after_attempt(5),
    reraise=True,
)

@gemini_retry
def embed_text(text):
    ## sending text to gemini's embedding model, and get back a vector representation
    result = client.models.embed_content(model="gemini-embedding-001", contents=text) ## 001 becuase its a text-only model, outputs 3072 dimension
    return result.embeddings[0].values ## take the first one [0], return .value (3072 floats)


if __name__ == "__main__":
    vec = embed_text("Apple stock rose sharply today.")
    print(f"Vector lenght: {len(vec)}") ## should be 3072
    print(f"First 5 values: {vec[:5]} ")
