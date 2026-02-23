import uuid, yaml
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

from config.settings import DatasetConfig
from config.datasets import OmniDLA

class Converter:
    def __init__(self, config: BaseModel):
        self.config = config
        self.path = self.config.path
        self.splits = self.config.splits
        self.class_names = self.config.class_names

    def prepare_path(self, split_name: str) -> None:
        self.labels_path = self.path / "labels" / split_name
        self.images_path = self.path / "images" / split_name
        self.mapped_labels_path = OmniDLA.path / "labels" / split_name
        self.mapped_images_path = OmniDLA.path / "images" / split_name

        for path in (self.labels_path, self.images_path, self.mapped_labels_path, self.mapped_images_path):
            path.mkdir(parents = True, exist_ok = True)

    def convert_image(self, image: Image.Image) -> str:
        self.data = []
        self.width, self.height = image.size
        self.image = image.convert("RGB").resize(DatasetConfig.image_size)

    def convert_bbox(self, bbox: tuple[float, float, float, float], category_id: int):
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / self.width
        y_center = (y_min + height / 2) / self.height
        width = width / self.width
        height = height / self.height
        self.data.append([x_center, y_center, width, height, category_id])

    def convert_labels(self):
        self.labels = []
        self.mapped_labels = []
        for x_center, y_center, width, height, category_id in self.data:
            self.labels.append(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            category_id = self.config.label_map[category_id]
            if category_id != -1:
                self.mapped_labels.append(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    def visual_data(self):
        self.visuals_path = self.path / "visuals"
        self.visuals_path.mkdir(parents = True, exist_ok = True)

        self.visual = self.image.copy()
        image_width, image_height = self.image.size

        drawer = ImageDraw.Draw(self.visual)
        text_font, text_color = ImageFont.load_default(size = 12), "red"
        
        for x_center, y_center, width, height, category_id in self.data:
            bbox_center = (x_center * image_width, y_center * image_height)
            bbox_size = (width * image_width, height * image_height)
            x_min, y_min = bbox_center[0] - bbox_size[0] / 2, bbox_center[1] - bbox_size[1] / 2
            x_max, y_max = bbox_center[0] + bbox_size[0] / 2, bbox_center[1] + bbox_size[1] / 2
            drawer.rectangle((x_min, y_min, x_max, y_max), outline = text_color, width = 1)
            drawer.text((x_min, y_min - 16), f"{self.class_names[category_id]}", fill = text_color, font = text_font)
        
    def save_data(self):
        data_id = str(uuid.uuid4())
        self.image.save(self.images_path / f"{data_id}.jpg", format = "JPEG", quality = 100)
        self.visual.save(self.visuals_path / f"{data_id}.png", format = "PNG", quality = 100)
        (self.labels_path / f"{data_id}.txt").write_text("\n".join(self.labels), encoding = "utf-8")

        self.image.save(self.mapped_images_path / f"{data_id}.jpg", format = "JPEG", quality = 100)
        (self.mapped_labels_path / f"{data_id}.txt").write_text("\n".join(self.mapped_labels), encoding = "utf-8")
        
    def save_yaml(self):
        self.path.mkdir(parents = True, exist_ok = True)
        data = {
            "path": str(self.path.resolve()),
            "names": self.class_names,
            **{split: f"images/{split}" for split in self.splits},
        }
        yaml_path = self.path / "dataset.yaml"
        with yaml_path.open("w", encoding = "utf-8") as file:
            yaml.safe_dump(data, file, allow_unicode = True, sort_keys = False)
