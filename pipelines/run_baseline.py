import sys
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    logging.info("Pipeline started")
    preprocess()
    train()
    accuracy, cv_accuracy = evaluate()
    logging.info("Accuracy: %.6f", accuracy)
    logging.info("CV accuracy: %.6f", cv_accuracy)
    logging.info("Pipeline completed")
    print("Baseline pipeline completed successfully.")
