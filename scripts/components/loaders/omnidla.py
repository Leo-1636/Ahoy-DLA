from components.converter import Converter
from config.datasets import OmniDLA

def load_omnidla():
    converter = Converter(OmniDLA)
    for split_name in OmniDLA.splits:
        converter.prepare_path(split_name)
        converter.save_yaml()

if __name__ == "__main__":
    load_omnidla()