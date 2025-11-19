from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DATASETS = ["adult", "credit"]
WIDTHS = [64, 128, 256]
DROPOUTS = [0.25, 0.50]
SEED = "1337"


def build_commands() -> list[list[str]]:
    commands: list[list[str]] = []
    for dataset in DATASETS:
        for width in WIDTHS:
            for dropout in DROPOUTS:
                # Baseline
                commands.append(
                    [
                        "--dataset",
                        dataset,
                        "--epochs",
                        "120",
                        "--patience",
                        "20",
                        "--width",
                        str(width),
                        "--dropout",
                        str(dropout),
                        "--no-dendrites",
                        "--seed",
                        SEED,
                        "--notes",
                        f"{dataset}_w{width}_d{dropout}_base",
                    ]
                )

                # Dendritic (DOING_HISTORY)
                dend_epochs = "600" if dataset == "credit" else "300"
                dend_patience = dend_epochs
                commands.append(
                    [
                        "--dataset",
                        dataset,
                        "--epochs",
                        dend_epochs,
                        "--patience",
                        dend_patience,
                        "--width",
                        str(width),
                        "--dropout",
                        str(dropout),
                        "--use-dendrites",
                        "--exclude-output-proj",
                        "--max-dendrites",
                        "8",
                        "--fixed-switch-num",
                        "50",
                        "--seed",
                        SEED,
                        "--notes",
                        f"{dataset}_w{width}_d{dropout}_dend",
                    ]
                )
    return commands


def main() -> None:
    train_script = ROOT / "train.py"
    for args in build_commands():
        cmd = [PYTHON, str(train_script), *args]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
