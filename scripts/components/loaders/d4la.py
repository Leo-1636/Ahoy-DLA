import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from components.converter import Converter
from config.datasets import D4LA

D4LA_ROOT = Path("D4LA")

def extract_data(json_data: dict) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    image_json = {image["id"]: image for image in json_data["images"]}
    element_json: dict[int, list[dict]] = {}
    for data in json_data["annotations"]:
        image_id = data["image_id"]
        if image_id not in element_json:
            element_json[image_id] = []
        element_json[image_id].append(data)
    return image_json, element_json

def load_d4la():
    converter = Converter(D4LA)
    for split_name in D4LA.splits:
        converter.prepare_path(split_name)

        json_path = D4LA_ROOT / f"{split_name}.json"
        with json_path.open("r", encoding = "utf-8") as file:
            json_data = json.load(file)
        image_json, element_json = extract_data(json_data)

        for image_id, image_data in tqdm(image_json.items(), desc=f"Processing {split_name} dataset"):
            try:
                image = Image.open(D4LA_ROOT / split_name / image_data["file_name"])
                converter.convert_image(image)

                for element_data in element_json.get(image_id, []):
                    converter.convert_bbox(tuple(element_data["bbox"]), element_data["category_id"])

                converter.convert_labels()
                converter.visual_data()
                converter.save_data()
                converter.save_yaml()
            except Exception as e:
                print(f"Error processing {split_name}: {e}")
                continue

if __name__ == "__main__":
    load_d4la()
