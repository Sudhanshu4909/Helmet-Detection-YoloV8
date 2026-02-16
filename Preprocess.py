
from pathlib import Path
from PIL import Image


DATA_PATH = Path("data/raw")
SPLITS = ["train", "valid", "test"]


def is_image_corrupt(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return False
    except Exception:
        return True


def clean_split(split):
    print(f"\nProcessing {split} dataset")

    image_dir = DATA_PATH / split / "images"
    label_dir = DATA_PATH / split / "labels"

    if not image_dir.exists():
        print(f"{split} split not found. Skipping.")
        return

    removed_corrupt = 0
    removed_missing_label = 0
    removed_empty_label = 0

    images = list(image_dir.glob("*.*"))

    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"

        # Remove corrupt images
        if is_image_corrupt(img_path):
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            removed_corrupt += 1
            continue

        # Remove images without labels
        if not label_path.exists():
            img_path.unlink(missing_ok=True)
            removed_missing_label += 1
            continue

        # Remove empty labels
        if label_path.stat().st_size == 0:
            img_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            removed_empty_label += 1

    total_images = len(list(image_dir.glob("*.*")))
    total_labels = len(list(label_dir.glob("*.txt")))

    print("Remaining images:", total_images)
    print("Remaining labels:", total_labels)
    print("Removed corrupt:", removed_corrupt)
    print("Removed missing labels:", removed_missing_label)
    print("Removed empty labels:", removed_empty_label)


def main():
    print("Starting dataset preprocessing")

    for split in SPLITS:
        clean_split(split)

    print("\nDataset preprocessing completed")


if __name__ == "__main__":
    main()
