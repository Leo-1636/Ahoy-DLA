import wandb as WandB
from ultralytics import YOLO

from scripts.config import wandb, train, model

WandB.login(key = wandb.api_key)

model = YOLO(model.yolo26)
model.train(
    data = "",
    epochs = train.epochs,
    imgsz = train.image_size,
    batch = train.batch_size,
    project = "Ahoy-RAG YOLO26",
    name = "train1",
)