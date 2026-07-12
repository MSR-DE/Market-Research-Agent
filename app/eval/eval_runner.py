from app.tools.rag_search import hybrid_search_reranked, search_chunk
from app.agent.orchestrator import run_agent
from app.ingestion.embedder import client, embed_text, gemini_retry
from app.ingestion.store import Session
from app.models import Chunk

# python -m app.eval.eval_runner

test_set = [
    {"question": "How did Apple's stock perform recently?", "expected_article_id": 10},
    {"question": "What did the Federal Reserve decide about interest rates?", "expected_article_id": 11},
    {"question": "What is Apple's lawsuit against OpenAI about?", "expected_article_id": 16},
    {"question": "What did Fed Governor Waller say about interest rate cuts?", "expected_article_id": 19},
    {"question": "How much could a $5,000 investment in SpaceX be worth by 2030?", "expected_article_id": 23},
    {"question": "What stocks should investors watch in Dow Jones futures trading?", "expected_article_id": 18},
    {"question": "What did a Federal Reserve official say about cutting interest rates?", "expected_article_id": 19},  # paraphrase test — no exact "Waller" or "Governor" terms
    {"question": "What was the CD account interest rate mentioned recently?", "expected_article_id": 20},  # specific number/term test — should favor keyword search
    {"question": "What was the 'LOL' moment mentioned in the OpenAI engineer story?", "expected_article_id": [13, 15]},  # duplicate articles — either counts as correct
    {"question": "What's going on between Apple and OpenAI?", "expected_article_id": 16},
    {"question": "4.10% APY", "expected_article_id": 20},
]

# separate list: questions with NO correct answer in the corpus.
# these can't be scored by ID-matching — they need eval_no_answer_cases() instead.
no_answer_set = [
    {"question": "What is Tesla's current stock price?"},
]


def eval_retrieval_baseline(test_set):
    # BASELINE: pure vector search, no hybrid, no RRF, no rerank
    correct = 0
    for case in test_set:
        results = search_chunk(case["question"], limit=3)
        retrieved_ids = [r.article_id for r in results]
        expected = case["expected_article_id"]

        if isinstance(expected, list):
            hit = any(e in retrieved_ids for e in expected)
        else:
            hit = expected in retrieved_ids
        correct += hit

        print(f"[{'PASS' if hit else 'FAIL'}] '{case['question']}' -> got {retrieved_ids}, expected {expected}")

    accuracy = correct / len(test_set)
    print(f"\nBaseline (vector-only) retrieval accuracy: {accuracy:.0%} ({correct}/{len(test_set)})")
    return accuracy


def eval_retrieval(test_set):  # measures RETRIEVAL QUALITY to check if search actually found the right article
    correct = 0
    for case in test_set:
        results = hybrid_search_reranked(case["question"], limit=3)
        retrieved_ids = [r.article_id for r in results]  ## what IDs actually came back
        expected = case["expected_article_id"]

        if isinstance(expected, list):
            hit = any(e in retrieved_ids for e in expected)
        else:
            hit = expected in retrieved_ids
        correct += hit  # True/False acts like 1/0 when added to a number

        print(f"[{'PASS' if hit else 'FAIL'}] '{case['question']}' -> got {retrieved_ids}, expected {expected}")

    accuracy = correct / len(test_set)
    print(f"\nRetrieval accuracy: {accuracy:.0%} ({correct}/{len(test_set)})")
    return accuracy


def eval_no_answer_cases(no_answer_set, distance_threshold=0.7):
    # tests whether the system correctly recognizes when NOTHING in the corpus is relevant,
    # using actual cosine distance instead of ID-matching (which can't express "nothing").
    correct = 0
    for case in no_answer_set:
        session = Session()
        query_vector = embed_text(case["question"])
        result = (
            session.query(Chunk, Chunk.embedding.cosine_distance(query_vector).label("dist"))
            .order_by("dist")
            .first()
        )
        session.close()

        best_distance = result.dist if result else 1.0
        correctly_flagged = best_distance > distance_threshold
        correct += correctly_flagged

        print(f"[{'PASS' if correctly_flagged else 'FAIL'}] '{case['question']}' -> closest distance: {best_distance:.3f} (threshold: {distance_threshold})")

    accuracy = correct / len(no_answer_set)
    print(f"\nNo-answer detection accuracy: {accuracy:.0%} ({correct}/{len(no_answer_set)})")
    return accuracy


###------------------------###

@gemini_retry   ## retries only on 429s with backoff — no blind sleeping
def _call_judge(prompt):
    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt)


def llm_judge(question, answer, source_chunks):  ## LLM-AS-JUDGE — scores the answer against its actual source material

    source_text = "\n".join(chunk.chunk_text for chunk in source_chunks)

    prompt = f"""You are evaluating an AI assistant's answer against its source material.

Question: {question}
Source material retrieved: {source_text}
Answer given: {answer}

Does the answer accurately reflect the source material? Rate 1-5
(5 = fully accurate and well-supported by the source; 1 = wrong or unsupported).
Respond with ONLY the number."""

    response = _call_judge(prompt)

    ## the judge is told to return only a number, but LLMs don't always comply
    try:
        return int(response.text.strip())
    except (ValueError, AttributeError):
        print(f"[Judge] Unparseable score: {response.text!r}")
        return None                              ## excluded from the average rather than silently miscounted


def eval_agent_answers(test_set):
    # runs the FULL agent, then judges the answer against what was actually retrieved
    scores = []
    for case in test_set:
        source_chunks = hybrid_search_reranked(case["question"])  # what the agent likely retrieved
        answer = run_agent(case["question"])                      # the agent's real answer

        score = llm_judge(case["question"], answer, source_chunks)
        if score is None:                        ## judge failed to return a usable score — skip, don't fake it
            print(f"[SKIP] '{case['question']}' — judge returned unparseable output")
            continue

        scores.append(score)
        print(f"[{score}/5] '{case['question']}'")

    avg = sum(scores) / len(scores)
    print(f"\nAverage answer quality: {avg:.1f}/5 (over {len(scores)} scored cases)")
    return avg


def check_judge_consistency(question, answer, source_chunks, runs=3):
    # runs the SAME judgment multiple times to see if the LLM judge is stable
    scores = [llm_judge(question, answer, source_chunks) for _ in range(runs)]
    print(f"Judge scores across {runs} runs: {scores}")
    return scores


if __name__ == "__main__":
    print("=== Baseline (Vector-Only) Eval ===")
    eval_retrieval_baseline(test_set)

    print("\n=== No-Answer Detection Eval ===")
    eval_no_answer_cases(no_answer_set)

    print("\n=== Retrieval Eval (Hybrid + RRF + Rerank) ===")
    eval_retrieval(test_set)

    print("\n=== Agent Answer Quality Eval ===")
    eval_agent_answers(test_set)