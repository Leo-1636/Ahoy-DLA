from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from components.converter import Converter
from config.datasets import DocSynth300K

def extract_data(string_datas: list, image: Image.Image) -> list[tuple[tuple[float, float, float, float], int]]:
    elements = []
    for string_data in string_datas:
        element_data = string_data.strip().split()
        category_id = int(element_data[0])
        bbox_data = [float(data) for data in element_data[1:]]

        x_min = min(bbox_data[0], bbox_data[2], bbox_data[4], bbox_data[6])
        y_min = min(bbox_data[1], bbox_data[3], bbox_data[5], bbox_data[7])
        x_max = max(bbox_data[0], bbox_data[2], bbox_data[4], bbox_data[6])
        y_max = max(bbox_data[1], bbox_data[3], bbox_data[5], bbox_data[7])
        elements.append(((x_min, y_min, (x_max - x_min), (y_max - y_min)), category_id))
    return elements

def load_docsynth():
    dataset = load_dataset(DocSynth300K.name)
    converter = Converter(DocSynth300K)
    for split_name in DocSynth300K.splits:
        converter.prepare_path(split_name)
        for data in tqdm(dataset[split_name], desc = f"Processing {split_name} dataset"):
            try:
                image = Image.open(BytesIO(data["image_data"]))
                converter.convert_image(image)

                for bbox, category_id in extract_data(data["anno_string"], image):
                    converter.convert_bbox(bbox, category_id)

                converter.convert_labels()
                converter.visual_data()
                converter.save_data()
            except Exception as e:
                print(f"Error processing {split_name} dataset: {e}")
                continue

if __name__ == "__main__":
    load_docsynth()