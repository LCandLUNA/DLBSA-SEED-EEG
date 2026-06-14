# main.py

from config import get_config
from preprocessing import preprocess_dataset
from training import run_experiment
from utils import create_folders


def main():
    config = get_config()
    create_folders(config)
    run_experiment(config)


if __name__ == "__main__":
    main()