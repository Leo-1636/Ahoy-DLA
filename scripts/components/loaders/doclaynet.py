from tqdm import tqdm
from datasets import load_dataset

from components.converter import Converter
from config.datasets import DocLayNet

def load_doclaynet():
    dataset = load_dataset(DocLayNet.name)
    converter = Converter(DocLayNet)
    for split_name in DocLayNet.splits:
        converter.prepare_path(split_name)
        for data in tqdm(dataset[split_name], desc = f"Processing {split_name} dataset"):
            try:
                converter.convert_image(data["image"])
                
                for bbox, category_id in zip(data["bboxes"], data["category_id"], strict = True):
                    converter.convert_bbox(bbox, category_id)
                
                converter.convert_labels()
                converter.visual_data()
                converter.save_data()
            except Exception as e:
                print(f"Error processing {split_name} dataset: {e}")
                continue

if __name__ == "__main__":
    load_doclaynet()
