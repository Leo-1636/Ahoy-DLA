import wandb as WandB
from ultralytics import YOLO

from config.settings import ModelConfig, TrainConfig, WandbConfig

WandB.login(key = WandbConfig.api_key)

model = YOLO(ModelConfig.yolo26)
model.train(
    data = "",
    epochs = TrainConfig.epochs,
    imgsz = TrainConfig.image_size,
    batch = TrainConfig.batch_size,
    project = "Ahoy-RAG YOLO26",
    name = "train1",
)