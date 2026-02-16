"""
Helmet Detection Model Training Script
Trains YOLOv8 model on helmet detection dataset
"""

from pathlib import Path
import yaml
from datetime import datetime
from ultralytics import YOLO
import torch


def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        print("GPU not available, using CPU")
        return 'cpu'


def load_config():
    """Load dataset configuration"""
    config_path = Path('data/raw/data.yaml')

    if not config_path.exists():
        raise FileNotFoundError(
            "Dataset configuration not found. Run dataset preparation first."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Classes: {config['names']}")
    return config


def train_model(
    model_size='n',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='cuda',
    pretrained=True
):

    model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
    model = YOLO(model_name)

    data_yaml = 'data/raw/data.yaml'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_name = 'runs/detect'
    run_name = f'helmet_detection_{timestamp}'

    print("Training started...")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project_name,
        name=run_name,
        patience=50,
        save=True,
        save_period=10,
        plots=True,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
    )

    print("Training completed")

    best_model_path = Path(project_name) / run_name / 'weights' / 'best.pt'
    final_model_path = Path('models') / f'helmet_detector_best_{timestamp}.pt'

    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model saved: {final_model_path}")

    return results, run_name


def main():
    device = check_gpu()

    try:
        load_config()
    except FileNotFoundError as e:
        print(e)
        return

    training_params = {
        'model_size': 'n',
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'device': device,
        'pretrained': True
    }

    response = input("Start training? (y/n): ").lower()
    if response != 'y':
        return

    try:
        results, run_name = train_model(**training_params)
        print(f"Results saved in runs/detect/{run_name}")
    except Exception as e:
        print("Training failed:", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
