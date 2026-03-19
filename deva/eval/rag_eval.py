# deva/eval/rag_eval.py
import json
import os
import pandas as pd

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from deva.config import LLM_PROVIDER, LLM_MODEL, EMBEDDINGS_PROVIDER, GEMINI_API_KEY
from deva.providers.llm import get_llm
from deva.providers.embeddings import get_embeddings
from deva.ingestion.indexer import get_or_create_vectorstore
from deva.app.rag_chain import create_rag_chain

load_dotenv()

TEST_DATASET_PATH = os.path.join(os.path.dirname(__file__), "test_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.csv")


def load_test_dataset():
    with open(TEST_DATASET_PATH, "r") as f:
        return json.load(f)


def run_pipeline_on_dataset(chain, test_data):
    """
    Run each question through the RAG chain and collect:
    - question (original)
    - answer (LLM output)
    - contexts (retrieved chunks as strings)
    - ground_truth (from test dataset)
    """
    results = []

    for item in test_data:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"▶ Running: {question}")
        result = chain.invoke({
            "question": question,
            "context_hint": "",
        })

        # Extract context strings from source docs
        contexts = [doc.page_content for doc in result.get("sources", [])]

        results.append({
            "user_input": question,
            "response": result["answer"],
            "retrieved_contexts": contexts,
            "reference": ground_truth,
        })

    return results


def build_ragas_dataset(results):
    return Dataset.from_list(results)


def run_evaluation():
    print("🔹 Loading vectorstore and chain...")
    vectorstore = get_or_create_vectorstore()
    llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    chain = create_rag_chain(vectorstore, llm)

    print("🔹 Loading test dataset...")
    test_data = load_test_dataset()

    print(f"🔹 Running {len(test_data)} questions through RAG pipeline...")
    results = run_pipeline_on_dataset(chain, test_data)

    print("🔹 Building RAGAS evaluation dataset...")
    ragas_dataset = build_ragas_dataset(results)

    # Use your existing LLM + embeddings as the RAGAS evaluator
    # so you don't need an OpenAI key
    evaluator_llm = LangchainLLMWrapper(
        get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        get_embeddings(provider=EMBEDDINGS_PROVIDER, api_key=GEMINI_API_KEY)
    )

    print("🔹 Running RAGAS evaluation...")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # Print scores
    print("\n📊 RAGAS Evaluation Results:")
    print("=" * 40)
    scores_df = result.to_pandas()
    print(scores_df.to_string(index=False))

    # Summary averages
    print("\n📈 Average Scores:")
    print("-" * 40)
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in scores_df.columns:
            print(f"  {col:25s}: {scores_df[col].mean():.4f}")

    # Save to CSV
    scores_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Results saved to {RESULTS_PATH}")

    return scores_df


if __name__ == "__main__":
    run_evaluation()
