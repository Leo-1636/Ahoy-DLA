import wandb as WandB
from ultralytics import YOLO

from config.settings import ModelConfig, TrainConfig, WandbConfig
from config.datasets import DocLayNet

WandB.login(key = WandbConfig.api_key)

model = YOLO(ModelConfig.yolo26)
model.train(
    data = DocLayNet.path / "dataset.yaml",
    epochs = TrainConfig.epochs,
    imgsz = TrainConfig.image_size,
    batch = TrainConfig.batch_size,
    project = "Ahoy-DLA",
    name = "train",
)