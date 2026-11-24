import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}",
)


def main() -> None:
    print("Hello from projeto-ml!")
