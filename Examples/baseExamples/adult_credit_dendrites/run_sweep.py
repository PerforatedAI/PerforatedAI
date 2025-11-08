"""
Convenience launcher for the Adult Income compression sweep.

Runs the baseline and three dendritic configurations.
Use `python run_sweep.py` or `make sweep`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

COMMANDS = [
    [
        "--epochs",
        "40",
        "--patience",
        "6",
        "--width",
        "512",
        "--dropout",
        "0.25",
        "--no-dendrites",
        "--notes",
        "baseline_w512",
    ],
    [
        "--epochs",
        "60",
        "--patience",
        "10",
        "--width",
        "128",
        "--dropout",
        "0.10",
        "--use-dendrites",
        "--exclude-output-proj",
        "--max-dendrites",
        "8",
        "--fixed-switch-num",
        "3",
        "--notes",
        "pai_w128_cap8",
    ],
    [
        "--epochs",
        "60",
        "--patience",
        "10",
        "--width",
        "128",
        "--dropout",
        "0.15",
        "--use-dendrites",
        "--exclude-output-proj",
        "--max-dendrites",
        "10",
        "--fixed-switch-num",
        "3",
        "--notes",
        "pai_w128_cap10_drop015",
    ],
    [
        "--epochs",
        "60",
        "--patience",
        "10",
        "--width",
        "128",
        "--dropout",
        "0.25",
        "--use-dendrites",
        "--exclude-output-proj",
        "--max-dendrites",
        "12",
        "--fixed-switch-num",
        "3",
        "--notes",
        "pai_w128_cap12",
    ],
]


def main() -> None:
    train_script = ROOT / "train.py"
    for args in COMMANDS:
        cmd = [PYTHON, str(train_script), *args]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
