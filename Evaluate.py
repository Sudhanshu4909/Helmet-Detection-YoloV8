
import os
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import json
from datetime import datetime
import pandas as pd


def find_latest_model():
    models_dir = Path('models')

    if not models_dir.exists():
        raise FileNotFoundError("No models directory found.")

    model_files = list(models_dir.glob('helmet_detector_best_*.pt'))

    if not model_files:
        raise FileNotFoundError("No trained models found.")

    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Using model: {latest_model}")

    return latest_model


def evaluate_model(model_path, data_yaml='data/raw/data.yaml'):
    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        split='val',
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        plots=True,
        save_json=True,
        save_hybrid=False,
    )

    return results


def calculate_metrics(results):
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
    }

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def plot_confusion_matrix(results, save_dir='results/evaluation'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        cm = results.confusion_matrix.matrix
        class_names = list(results.names.values())

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=class_names + ['Background'],
                    yticklabels=class_names + ['Background'],
                    ax=ax1)
        ax1.set_title('Confusion Matrix')

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                    xticklabels=class_names + ['Background'],
                    yticklabels=class_names + ['Background'],
                    ax=ax2)
        ax2.set_title('Normalized Confusion Matrix')

        plt.tight_layout()

        cm_path = Path(save_dir) / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved: {cm_path}")


def generate_evaluation_report(metrics, model_path, save_dir='results/evaluation'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""
Model Path: {model_path}
Evaluation Date: {timestamp}

mAP@0.5: {metrics['mAP50']:.4f}
mAP@0.5:0.95: {metrics['mAP50-95']:.4f}
Precision: {metrics['Precision']:.4f}
Recall: {metrics['Recall']:.4f}
"""

    report_path = Path(save_dir) / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    metrics_path = Path(save_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Report saved: {report_path}")
    print(f"Metrics saved: {metrics_path}")

    return report


def main():
    try:
        model_path = find_latest_model()

        results = evaluate_model(model_path)

        metrics = calculate_metrics(results)

        plot_confusion_matrix(results)
        generate_evaluation_report(metrics, model_path)

        print("Evaluation complete. Results saved in results/evaluation/")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
