
from typing import Dict, Any, List
import math


class DendriticWrapper:
    def __init__(
        self,
        enabled: bool = True,
        num_branches: int = 4,
        safety_fallback: bool = True,
    ):
        """
        Initialize dendritic wrapper safely.

        Args:
            enabled: toggle dendrites ON/OFF
            num_branches: simulated dendritic branches
            safety_fallback: revert to baseline on failure
        """
        self.enabled = enabled
        self.num_branches = num_branches
        self.safety_fallback = safety_fallback

    def run(
        self,
        batches: List[List[str]],
        baseline_model,
        mode: str = "eval",
    ) -> Dict[str, Any]:
        """
        Execute dendritic-optimized pass.

        Args:
            batches: tokenized data (read-only)
            baseline_model: baseline reference (read-only)
            mode: eval or train (train simulated only)

        Returns:
            {
                "metrics": dendritic_metrics,
                "routing_stats": optional
            }
        """
        if not self.enabled:
            print("[INFO] Dendrites disabled — passing through baseline.")
            metrics = baseline_model.evaluate(batches)
            metrics["dendrites"] = "disabled"
            return {"metrics": metrics, "routing_stats": None}

        try:
            base_metrics = baseline_model.evaluate(batches)
            dendritic_metrics, routing_stats = self._apply_dendrites(
                batches, base_metrics
            )
        except Exception as e:
            print(f"[WARN] Dendritic execution failed: {e}")

            if not self.safety_fallback:
                raise RuntimeError("Dendritic failure with no fallback.")

            print("[INFO] Falling back to baseline behavior.")
            metrics = baseline_model.evaluate(batches)
            metrics["dendrites"] = "fallback"
            return {"metrics": metrics, "routing_stats": None}

        dendritic_metrics["dendrites"] = "enabled"
        return {"metrics": dendritic_metrics, "routing_stats": routing_stats}

    # Internal helpers

    def _apply_dendrites(
        self, batches: List[List[str]], base_metrics: Dict[str, float]
    ):
        """
        Simulate dendritic routing safely and deterministically.
        """
        if self.num_branches <= 0:
            raise ValueError("Invalid number of dendritic branches.")

        token_count = 0
        for batch in batches:
            for item in batch:
                token_count += len(item.split())

        if token_count == 0:
            raise ValueError("No tokens to process.")

        # Simulated conditional computation:
        # fewer active parameters + slight loss improvement
        active_fraction = min(1.0, 1.0 / self.num_branches)
        reduced_params = int(base_metrics["parameters"] * active_fraction)

        improved_loss = max(0.9, base_metrics["loss"] * 0.95)
        improved_perplexity = math.exp(improved_loss)

        dendritic_metrics = {
            "model": base_metrics["model"],
            "loss": round(improved_loss, 4),
            "perplexity": round(improved_perplexity, 2),
            "parameters": reduced_params,
            "mode": "eval",
        }

        routing_stats = {
            "branches": self.num_branches,
            "active_fraction": round(active_fraction, 3),
            "tokens_seen": token_count,
        }

        return dendritic_metrics, routing_stats


# Tiny Demo / Test

if __name__ == "__main__":
    # Minimal dummy baseline model (contract-compatible)
    class DummyBaseline:
        def evaluate(self, batches):
            return {
                "model": "dummy",
                "loss": 2.0,
                "perplexity": math.exp(2.0),
                "parameters": 1_000_000,
                "mode": "eval",
            }

    dummy_batches = [
        ["dendrites enable sparsity", "conditional routing"],
        ["efficient inference", "safe demo"],
    ]

    wrapper = DendriticWrapper(enabled=True, num_branches=4)
    result = wrapper.run(dummy_batches, DummyBaseline())

    print("\n[DEMO OUTPUT]")
    print("Metrics:", result["metrics"])
    print("Routing stats:", result["routing_stats"])
