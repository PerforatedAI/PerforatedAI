
from typing import Dict, Any


class MetricsViewer:
    def __init__(self, display_mode: str = "auto"):
        """
        Initialize viewer safely.

        Args:
            display_mode:
                - 'auto'   : try table + plot
                - 'table'  : terminal only
                - 'static' : fallback static output
        """
        self.display_mode = display_mode

    def show(
        self,
        baseline_metrics: Dict[str, Any],
        dendritic_metrics: Dict[str, Any],
    ) -> None:
        """
        Display metrics safely.

        Args:
            baseline_metrics: reference metrics
            dendritic_metrics: dendritic metrics
        """
        try:
            self._print_table(baseline_metrics, dendritic_metrics)
        except Exception as e:
            print(f"[WARN] Table display failed: {e}")
            print("[INFO] Switching to static display.")
            self._print_static(baseline_metrics, dendritic_metrics)

    # -----------------------
    # Internal helpers
    # -----------------------

    def _print_table(
        self,
        baseline: Dict[str, Any],
        dendritic: Dict[str, Any],
    ) -> None:
        """
        Print clean side-by-side comparison.
        """
        print("\n=== MODEL COMPARISON (Baseline vs Dendritic) ===")

        keys = ["loss", "perplexity", "parameters"]
        for key in keys:
            b_val = baseline.get(key, "N/A")
            d_val = dendritic.get(key, "N/A")

            print(f"{key:>12} | baseline: {b_val:<10} | dendritic: {d_val:<10}")

        print("\nStatus:")
        print(f"  Baseline mode  : {baseline.get('mode', 'unknown')}")
        print(f"  Dendrites      : {dendritic.get('dendrites', 'unknown')}")

    def _print_static(
        self,
        baseline: Dict[str, Any],
        dendritic: Dict[str, Any],
    ) -> None:
        """
        Ultra-safe fallback output.
        """
        print("\n[STATIC SUMMARY]")
        print(f"Baseline loss      : {baseline.get('loss')}")
        print(f"Dendritic loss     : {dendritic.get('loss')}")
        print(f"Baseline params    : {baseline.get('parameters')}")
        print(f"Dendritic params   : {dendritic.get('parameters')}")
        print("Demo continued safely.")


# -----------------------
# Tiny Demo / Test
# -----------------------
if __name__ == "__main__":
    baseline_metrics = {
        "loss": 2.0,
        "perplexity": 7.39,
        "parameters": 1_000_000,
        "mode": "eval",
    }

    dendritic_metrics = {
        "loss": 1.9,
        "perplexity": 6.69,
        "parameters": 250_000,
        "mode": "eval",
        "dendrites": "enabled",
    }

    viewer = MetricsViewer()
    viewer.show(baseline_metrics, dendritic_metrics)
