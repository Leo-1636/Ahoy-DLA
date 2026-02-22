from components.loaders.d4la import load_d4la
from components.loaders.doclaynet import load_doclaynet
from components.loaders.docsynth import load_docsynth

def load_datasets():
    load_d4la()
    load_doclaynet()
    load_docsynth()

if __name__ == "__main__":
    load_datasets()