import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import argparse


def find_latest_model():
    models_dir = Path('models')

    if not models_dir.exists():
        raise FileNotFoundError("No models directory found.")

    model_files = list(models_dir.glob('helmet_detector_best_*.pt'))

    if not model_files:
        raise FileNotFoundError("No trained models found.")

    return max(model_files, key=os.path.getmtime)


def run_inference_image(model, image_path, conf_threshold=0.25, save_dir='results/inference'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=0.5,
        save=False,
        verbose=False
    )[0]

    annotated_image = results.plot()

    output_name = f"{Path(image_path).stem}_detected.jpg"
    output_path = Path(save_dir) / output_name

    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), annotated_bgr)

    print(f"{Path(image_path).name}: {len(results.boxes)} detections")
    print(f"Saved: {output_path}")

    return output_path, results


def run_inference_video(model, video_path, conf_threshold=0.25, save_dir='results/inference'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_name = f"{Path(video_path).stem}_detected.mp4"
    output_path = Path(save_dir) / output_name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=0.5,
            verbose=False
        )[0]

        annotated_frame = results.plot()
        out.write(annotated_frame)

        detection_count += len(results.boxes)
        frame_count += 1

    cap.release()
    out.release()

    avg_detections = detection_count / frame_count if frame_count else 0

    print(f"Video processed: {frame_count} frames")
    print(f"Avg detections/frame: {avg_detections:.2f}")
    print(f"Saved: {output_path}")

    return output_path


def run_inference_webcam(model, conf_threshold=0.25):
    cap = cv2.VideoCapture(0)

    print("Webcam running. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=0.5,
            verbose=False
        )[0]

        annotated_frame = results.plot()
        cv2.imshow('Helmet Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def batch_inference(model, input_dir, conf_threshold=0.25, save_dir='results/inference'):
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    results_list = []

    for image_path in image_files:
        output_path, results = run_inference_image(
            model, image_path, conf_threshold, save_dir
        )
        results_list.append({
            'image': image_path.name,
            'detections': len(results.boxes),
            'output': output_path
        })

    print(f"Processed {len(image_files)} images")
    return results_list


def main():
    parser = argparse.ArgumentParser(description='Helmet Detection Inference')
    parser.add_argument('--source', type=str)
    parser.add_argument('--type', choices=['image', 'video', 'webcam', 'batch'], default='image')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()

    try:
        model_path = args.model if args.model else find_latest_model()
        model = YOLO(model_path)

        if args.type == 'image':
            run_inference_image(model, args.source, args.conf)

        elif args.type == 'video':
            run_inference_video(model, args.source, args.conf)

        elif args.type == 'webcam':
            run_inference_webcam(model, args.conf)

        elif args.type == 'batch':
            batch_inference(model, args.source, args.conf)

    except Exception as e:
        print("Inference failed:", e)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("Run with arguments. Example:")
        print("python 04_inference.py --type image --source img.jpg")
