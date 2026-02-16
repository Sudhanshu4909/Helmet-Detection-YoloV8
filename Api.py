from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager


model = None
model_info = {}


def find_latest_model():
    models_dir = Path("models")
    if not models_dir.exists():
        return None

    model_files = list(models_dir.glob("helmet_detector_best_*.pt"))
    if not model_files:
        return None

    return max(model_files, key=lambda p: p.stat().st_mtime)


def load_model(model_path: str = None):
    global model, model_info

    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            raise FileNotFoundError("No trained model found.")

    model = YOLO(model_path)
    model_info = {
        "model_path": str(model_path),
        "loaded_at": datetime.now().isoformat(),
        "model_size": Path(model_path).stat().st_size / (1024 * 1024),
    }

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
        print(f"Model loaded: {model_info['model_path']}")
    except Exception as e:
        print(f"Model not loaded: {e}")
    yield


app = FastAPI(
    title="Helmet Detection API",
    description="REST API for helmet detection using YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]


class PredictionResponse(BaseModel):
    success: bool
    detections: List[DetectionResult]
    image_size: List[int]
    inference_time: float
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    model_info: dict


@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "message": "Helmet Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_annotated": "/predict/annotated",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_path": model_info.get("model_path"),
        "model_info": model_info,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    height, width = image.shape[:2]

    start_time = datetime.now()
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )[0]
    inference_time = (datetime.now() - start_time).total_seconds()

    detections = [
        DetectionResult(
            class_name=results.names[int(box.cls[0])],
            confidence=float(box.conf[0]),
            bbox=box.xyxy[0].tolist(),
        )
        for box in results.boxes
    ]

    return PredictionResponse(
        success=True,
        detections=detections,
        image_size=[width, height],
        inference_time=inference_time,
        message=f"{len(detections)} object(s) detected",
    )


@app.post("/predict/annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )[0]

    annotated_image = results.plot()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, annotated_image)
        tmp_path = tmp.name

    return FileResponse(tmp_path, media_type="image/jpeg")


def main():
    print("Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
