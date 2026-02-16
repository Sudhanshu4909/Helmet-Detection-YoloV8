import os
from pathlib import Path
import shutil
from roboflow import Roboflow


def setup_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Directories created")


def download_dataset():
    """
    Download helmet dataset using Roboflow
    """

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        print("ROBOFLOW_API_KEY not found.")
        choice = input("Create sample dataset structure instead? (y/n): ").lower()
        if choice == "y":
            create_sample_structure()
        return

    try:
        print("Downloading dataset from Roboflow...")

        rf = Roboflow(api_key=api_key)
        project = rf.workspace("helmet-detection-cnnrq").project("helmet-detection-miti4")
        version = project.version(1)

        dataset = version.download("yolov8")

        src_path = Path(dataset.location)
        dst_path = Path("data/raw")

        # Move dataset into data/raw
        for item in src_path.iterdir():
            target = dst_path / item.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(item), str(target))

        print("Dataset downloaded and moved to data/raw")

    except Exception as e:
        print("Dataset download failed:", e)
        choice = input("Create sample dataset structure instead? (y/n): ").lower()
        if choice == "y":
            create_sample_structure()


def create_sample_structure():
    """Create sample dataset structure"""

    base_path = Path('data/raw')

    for split in ['train', 'valid', 'test']:
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    yaml_content = """names:
  0: helmet
  1: no-helmet
  2: person

nc: 3

train: train/images
val: valid/images
test: test/images
"""

    with open(base_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print("Sample dataset structure created")


def verify_dataset():
    """Verify dataset structure"""

    base_path = Path('data/raw')
    required_paths = [
        base_path / 'train' / 'images',
        base_path / 'train' / 'labels',
        base_path / 'valid' / 'images',
        base_path / 'valid' / 'labels',
        base_path / 'data.yaml'
    ]

    all_exist = True
    for path in required_paths:
        if not path.exists():
            print(f"Missing: {path}")
            all_exist = False

    if all_exist:
        train_images = len(list((base_path / 'train' / 'images').glob('*.*')))
        valid_images = len(list((base_path / 'valid' / 'images').glob('*.*')))

        if train_images > 0 and valid_images > 0:
            print("Dataset ready for training")
            return True
        else:
            print("Dataset structure exists but images missing")
            return False
    else:
        print("Dataset structure incomplete")
        return False


def main():
    print("Helmet Detection Dataset Preparation")

    setup_directories()
    download_dataset()
    dataset_ready = verify_dataset()

    if dataset_ready:
        print("Setup complete. Proceed to training.")
    else:
        print("Setup incomplete. Add dataset and rerun.")


if __name__ == "__main__":
    main()
