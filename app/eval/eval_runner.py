import time
from app.tools.rag_search import hybrid_search_reranked
from app.agent.orchestrator import run_agent
from app.ingestion.embedder import client

test_set = [
{"question": "How did Apple's stock perform recently?", "expected_article_id": 10},
{"question": "What did the Federal Reserve decide about interest rates?", "expected_article_id": 11},
{"question": "What is Apple's lawsuit against OpenAI about?", "expected_article_id": 16},
{"question": "What did Fed Governor Waller say about interest rate cuts?", "expected_article_id": 19},
{"question": "How much could a $5,000 investment in SpaceX be worth by 2030?", "expected_article_id": 23},
{"question": "What stocks should investors watch in Dow Jones futures trading?", "expected_article_id": 18},
]


def eval_retrieval(test_set): # measures RETRIEVAL QUALITY to check if did search actually find the right article
    correct = 0
    for case in test_set:
        results = hybrid_search_reranked(case["question"], limit=3)
        retrieved_ids = [r.article_id for r in results]    ## what IDs actually came back
        hit = case["expected_article_id"] in retrieved_ids
        correct += hit     # True/False acts like 1/0 when added to a number

        print(f"[{'PASS' if hit else 'FAIL'}] '{case['question']}' -> got {retrieved_ids}, expected {case['expected_article_id']}")

    accuracy = correct / len(test_set)
    print(f"\nRetrieval accuracy: {accuracy:.0%} ({correct}/{len(test_set)})")
    return accuracy
    


###------------------------###

def llm_judge(question, answer, source_chunks): ## Uses LLM to score quality  - "LLM-AS-JUDGE" - LLM's own judgement based on the prompt
    
    source_text = "\n".join(chunk.chunk_text for chunk in source_chunks)

    prompt = f"""You are evaluating an AI assistant's answer against its source material.

Question: {question}
Source material retrieved: {source_text}
Answer given: {answer}

Does the answer accurately reflect the source material? Rate 1-5
(5 = fully accurate and well-supported by the source; 1 = wrong or unsupported).
Respond with ONLY the number."""

    time.sleep(20) ## extra pause right before the call, since this is a SEPARATE rate-limited request from the ones inside run_agent()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return int(response.text.strip())



def eval_agent_answers(test_set):
    # runs the FULL agent, then judges the answer against what was actually retrieved
    scores = []
    for case in test_set:
        source_chunks = hybrid_search_reranked(case["question"])  # what the agent likely retrieved
        answer = run_agent(case["question"])                      # the agent's real answer

        score = llm_judge(case["question"], answer, source_chunks)
        scores.append(score)
        print(f"[{score}/5] '{case['question']}'")

        time.sleep(15) ## kept hitting rate-limit 

    avg = sum(scores) / len(scores)
    print(f"\nAverage answer quality: {avg:.1f}/5")
    return avg


if __name__ == "__main__":
    print("=== Retrieval Eval ===")
    eval_retrieval(test_set)

    print("\n=== Agent Answer Quality Eval ===")
    eval_agent_answers(test_set)