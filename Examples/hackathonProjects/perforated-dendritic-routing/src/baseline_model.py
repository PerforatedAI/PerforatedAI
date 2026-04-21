

from typing import Dict, Any, List
import math


class BaselineModel:
    def __init__(
        self,
        model_name: str = "demo_baseline",
        pretrained: bool = True,
        eval_only: bool = True,
    ):
        """
        Initialize baseline model safely.

        Args:
            model_name: logical name only
            pretrained: flag for clarity (no downloads here)
            eval_only: enforce inference-only behavior
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.eval_only = eval_only

        # Simulated model capacity (fixed)
        self.param_count = 1_000_000

    def evaluate(self, batches: List[List[str]]) -> Dict[str, float]:
        """
        Run baseline evaluation.

        Args:
            batches: tokenized batches (read-only)

        Returns:
            baseline_metrics: dict of reference metrics
        """
        try:
            loss = self._compute_dummy_loss(batches)
            perplexity = math.exp(loss)
        except Exception as e:
            print(f"[WARN] Baseline evaluation failed: {e}")
            print("[INFO] Falling back to safe default metrics.")

            # Safe fallback values
            loss = 5.0
            perplexity = math.exp(loss)

        return {
            "model": self.model_name,
            "loss": round(loss, 4),
            "perplexity": round(perplexity, 2),
            "parameters": self.param_count,
            "mode": "eval_only",
        }

    # Internal helpers

    def _compute_dummy_loss(self, batches: List[List[str]]) -> float:
        """
        Deterministic, safe loss proxy.
        NEVER depends on randomness or external state.
        """
        if not batches or not batches[0]:
            raise ValueError("Empty batch input.")

        token_count = 0
        for batch in batches:
            for item in batch:
                token_count += len(item.split())

        # Simple deterministic proxy loss
        return max(1.0, 10.0 / max(token_count, 1))


# Tiny Demo / Test

if __name__ == "__main__":
    dummy_batches = [
        ["dendritic routing improves efficiency", "baseline model reference"],
        ["conditional computation saves cost", "simple demo batch"],
    ]

    model = BaselineModel()
    metrics = model.evaluate(dummy_batches)

    print("\n[DEMO OUTPUT]")
    for k, v in metrics.items():
        print(f"{k}: {v}")
