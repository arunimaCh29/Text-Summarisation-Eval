from config import Config
from data.dataset_loader import DataSetLoader


def main():
   
    loader = DataSetLoader()
    loader.load(debug_mode=False,batch_size=128)

if __name__ == "__main__":
    main()