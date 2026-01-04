
import os
import random
from typing import List, Dict, Any


class DatasetLoader:
    def __init__(
        self,
        dataset_name: str = "demo_text",
        max_samples: int = 32,
        seed: int = 42,
        cache_dir: str = "./cached_data",
    ):
        """
        Initialize dataset loader with SAFE defaults.

        Args:
            dataset_name: logical name only (no downloading!)
            max_samples: fixed small size for demo
            seed: ensures determinism
            cache_dir: local-only storage
        """
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.seed = seed
        self.cache_dir = cache_dir

        # Enforce deterministic behavior
        random.seed(self.seed)

    def load(self) -> Dict[str, Any]:
        """
        Load dataset safely.

        Returns:
            {
                "batches": List[List[str]],
                "metadata": Dict
            }
        """
        try:
            data = self._load_from_cache()
        except Exception as e:
            print(f"[WARN] Cache load failed: {e}")
            print("[INFO] Falling back to built-in demo dataset.")
            data = self._fallback_dataset()

        batches = self._make_batches(data)

        return {
            "batches": batches,
            "metadata": {
                "dataset": self.dataset_name,
                "num_samples": len(data),
                "batch_size": len(batches[0]) if batches else 0,
                "seed": self.seed,
                "source": "cache" if self._cache_exists() else "fallback",
            },
        }

    # Internal helpers

    def _cache_exists(self) -> bool:
        return os.path.exists(self._cache_path())

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.dataset_name}.txt")

    def _load_from_cache(self) -> List[str]:
        """
        Load dataset from local cache ONLY.
        """
        path = self._cache_path()
        if not os.path.exists(path):
            raise FileNotFoundError("Cached dataset not found.")

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError("Cached dataset is empty.")

        return lines[: self.max_samples]

    def _fallback_dataset(self) -> List[str]:
        """
        Guaranteed-safe fallback dataset.
        NEVER fails.
        """
        demo_data = [
            "Dendritic optimization enables conditional computation.",
            "Not all inputs need full model activation.",
            "Sparse routing improves efficiency.",
            "Biological neurons inspire modern AI.",
            "Efficient models reduce inference cost.",
        ]

        # Repeat deterministically if needed
        while len(demo_data) < self.max_samples:
            demo_data += demo_data

        return demo_data[: self.max_samples]

    def _make_batches(self, data: List[str], batch_size: int = 4) -> List[List[str]]:
        """
        Create fixed-shape batches.
        """
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            # Enforce fixed batch size by padding
            if len(batch) < batch_size:
                batch += [""] * (batch_size - len(batch))

            batches.append(batch)

        return batches


# Tiny Demo / Test

if __name__ == "__main__":
    loader = DatasetLoader(max_samples=8)
    result = loader.load()

    print("\n[DEMO OUTPUT]")
    print("Metadata:", result["metadata"])
    print("First batch:", result["batches"][0])
