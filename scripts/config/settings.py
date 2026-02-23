
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    yolo26 = "yolo26l.pt"
    yolo11 = "yolo11l.pt"

class TrainConfig:
    epochs = 50
    batch_size = 16
    image_size = 640

class DatasetConfig:
    path: Path = Path(__file__).parents[2] / "datasets"
    image_size: tuple[int, int] = (640, 640)

class WandbConfig:
    api_key: str | None = os.getenv("WANDB_API_KEY")
