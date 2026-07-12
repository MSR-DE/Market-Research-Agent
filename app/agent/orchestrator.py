from google.genai import types
from app.ingestion.embedder import client, gemini_retry
from app.tools.rag_search import hybrid_search_reranked


search_tool_declaration = {
    "name": "search_news",
    "description": "Search stored financial news articles for information relevant to a query. Use this when you need factual information about companies, markets, or events to answer the user's question.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, e.g. 'Apple earnings' or 'Federal Reserve interest rates'"
            }
        },
        "required": ["query"]
    }
}

tool = types.Tool(function_declarations=[search_tool_declaration])

# maps tool NAME (as the model will refer to it) to the ACTUAL function that runs it
available_tools = {
    "search_news": lambda query: hybrid_search_reranked(query)
}


@gemini_retry   ## retries ONLY on 429s, with exponential backoff
def _call_model(contents):
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[tool])
    )


def run_agent(user_query, max_iterations=5):
    contents = [{"role": "user", "parts": [{"text": user_query}]}]

    for i in range(max_iterations):
        try:
            response = _call_model(contents)
        except Exception as e:
            print(f"[Agent] Gemini unavailable ({e}), falling back to raw search")
            from app.tools.rag_search import search_chunk
            fallback_results = search_chunk(user_query, limit=3)
            fallback_text = "\n\n".join(chunk.chunk_text for chunk in fallback_results)
            return f"AI reasoning unavailable right now. Here's the most relevant raw content I found:\n\n{fallback_text}"

        part = response.candidates[0].content.parts[0]

        # CASE 1: the model wants to call a tool
        if part.function_call:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)

            print(f"[Agent] Calling tool: {fn_name}({fn_args})")

            if fn_name not in available_tools:
                tool_result_text = f"Error: tool '{fn_name}' does not exist."
            else:
                try:
                    tool_result = available_tools[fn_name](**fn_args)
                    tool_result_text = "\n".join(chunk.chunk_text for chunk in tool_result)
                except Exception as e:
                    tool_result_text = f"Error running tool: {str(e)}"

            # add the model's tool-call request AND the tool's result to the conversation
            contents.append(response.candidates[0].content)
            contents.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": fn_name,
                        "response": {"result": tool_result_text}
                    }
                }]
            })
            # loop again, model gets another turn, now with the search results

        # CASE 2: the model gave a final answer, no tool call needed
        else:
            return part.text

    return "Reached max iterations without a final answer."


if __name__ == "__main__":
    answer = run_agent("How is Apple's stock performing based on recent news?")
    print("\nFinal answer:", answer)