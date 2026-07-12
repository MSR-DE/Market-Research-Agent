from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_api_key)

def embed_text(text):
    ## sending text to gemini's embedding model, and get back a vector representation
    result = client.models.embed_content(model="gemini-embedding-001", contents=text) ## 001 becuase its a text-only model, outputs 3072 dimension
    return result.embeddings[0].values ## take the first one [0], return .value (3072 floats)


if __name__ == "__main__":
    vec = embed_text("Apple stock rose sharply today.")
    print(f"Vector lenght: {len(vec)}") ## should be 3072
    print(f"First 5 values: {vec[:5]} ")
