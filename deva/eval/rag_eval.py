"""
RAGAS evaluation script.

Usage:
    python -m deva.eval.rag_eval

Expects deva/eval/test_dataset.json to contain Q&A pairs.
Results saved to deva/eval/eval_results.csv.
"""
import json
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from deva.config import LLM_PROVIDER, LLM_MODEL, EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL, GEMINI_API_KEY
from deva.providers.llm import get_llm
from deva.providers.embeddings import get_embeddings
from deva.app.graph import ask as graph_ask
from deva.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DIR            = os.path.dirname(__file__)
DATASET_PATH   = os.path.join(DIR, "test_dataset.json")
RESULTS_PATH   = os.path.join(DIR, "eval_results.csv")


def load_test_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def run_pipeline(test_data: list) -> list:
    results = []
    for item in test_data:
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")
        logger.info(f"Evaluating: {question!r}")

        result   = graph_ask(question=question)
        contexts = [doc.page_content for doc in (result.get("sources") or [])]

        results.append({
            "user_input":         question,
            "response":           result.get("answer", ""),
            "retrieved_contexts": contexts,
            "reference":          ground_truth,
        })
    return results


def run_evaluation():
    logger.info("Loading test dataset...")
    test_data = load_test_dataset()
    logger.info(f"Running {len(test_data)} questions through pipeline...")
    results = run_pipeline(test_data)

    ragas_dataset = Dataset.from_list(results)

    eval_llm = LangchainLLMWrapper(
        get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    )
    eval_emb = LangchainEmbeddingsWrapper(
        get_embeddings(provider=EMBEDDINGS_PROVIDER, model=EMBEDDINGS_MODEL)
    )

    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=eval_llm,
        embeddings=eval_emb,
    )

    df = result.to_pandas()
    print("\n📊 RAGAS Results:")
    print("=" * 50)
    print(df.to_string(index=False))
    print("\n📈 Averages:")
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            print(f"  {col:<25}: {df[col].mean():.4f}")

    df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Results saved to {RESULTS_PATH}")
    return df


if __name__ == "__main__":
    run_evaluation()
