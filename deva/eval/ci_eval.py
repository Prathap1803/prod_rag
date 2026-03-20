"""
CI evaluation gate — fails with exit code 1 if any metric is below threshold.
"""
import sys
from deva.eval.rag_eval import run_evaluation
from deva.logger import get_logger

logger = get_logger(__name__)

THRESHOLDS = {
    "faithfulness":      0.75,
    "answer_relevancy":  0.70,
    "context_precision": 0.65,
    "context_recall":    0.60,
}

if __name__ == "__main__":
    df     = run_evaluation()
    failed = []

    for metric, threshold in THRESHOLDS.items():
        if metric in df.columns:
            avg = df[metric].mean()
            if avg < threshold:
                failed.append(f"{metric}: {avg:.4f} < {threshold}")

    if failed:
        print("\n❌ CI FAILED — metrics below threshold:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\n✅ CI PASSED — all metrics above threshold")
        sys.exit(0)
