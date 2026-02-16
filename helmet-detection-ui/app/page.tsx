"use client";

import React, { useEffect, useRef, useState } from "react";

type Detection = {
  class_name: string;
  confidence: number;
  bbox: number[]; // [x1,y1,x2,y2]
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://sudhanshu03-helmet-detection.hf.space";

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Cleanup preview URL when component unmounts
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // When previewUrl becomes null, clear canvas and detections only if needed.
  useEffect(() => {
    if (!previewUrl) {
      // only clear if we actually had detections to avoid re-renders
      setDetections((prev) => (prev.length ? [] : prev));
      clearCanvas();
      setError(null);
    }
    // note: we intentionally do NOT include `detections` here to avoid loops
  }, [previewUrl]);

  // When detections change, draw them (no state updates here)
  useEffect(() => {
    if (!previewUrl) return;
    // drawBoxes will read detections from state; it does not set state.
    drawBoxes();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detections, previewUrl]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const f = e.target.files?.[0] ?? null;
    if (!f) {
      // revoke old preview if any
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      setFile(null);
      setPreviewUrl(null);
      setDetections([]);
      return;
    }
    // revoke previous preview
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const url = URL.createObjectURL(f);
    setFile(f);
    setPreviewUrl(url);
    setDetections([]); // user-initiated clear is fine
  };

  // Programmatically open file picker
  const openFilePicker = () => {
    setError(null);
    inputRef.current?.click();
  };

  const callPredict = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setDetections([]);

    try {
      const fd = new FormData();
      fd.append("file", file);

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`Server returned ${res.status} ${text}`);
      }

      const json = await res.json();

      const dets: Detection[] = (json.detections ?? []).map((d: any) => ({
        class_name: String(d.class_name ?? d.class ?? d.label ?? "unknown"),
        confidence: Number(d.confidence ?? d.conf ?? 0),
        bbox: Array.isArray(d.bbox) ? d.bbox.map(Number) : [],
      }));

      setDetections(dets);
      // ensure drawing happens after image fully loads — img onLoad also triggers draw
    } catch (err: any) {
      setError(err?.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  // drawBoxes uses detections and image refs, never mutates state
  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawBoxes() {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    // compute displayed size
    const rect = img.getBoundingClientRect();
    // Use integer canvas size
    canvas.width = Math.round(rect.width);
    canvas.height = Math.round(rect.height);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // scale from natural image size to displayed size
    const naturalW = img.naturalWidth || 1;
    const naturalH = img.naturalHeight || 1;
    const scaleX = canvas.width / naturalW;
    const scaleY = canvas.height / naturalH;

    ctx.lineWidth = Math.max(2, Math.floor(canvas.width / 300));
    ctx.font = `${Math.max(12, Math.floor(canvas.width / 40))}px Inter, Arial`;

    detections.forEach((d) => {
      if (!d.bbox || d.bbox.length < 4) return;
      const [x1, y1, x2, y2] = d.bbox.map(Number);
      const x = x1 * scaleX;
      const y = y1 * scaleY;
      const w = (x2 - x1) * scaleX;
      const h = (y2 - y1) * scaleY;

      const cls = d.class_name.toLowerCase();
      const color = cls.includes("helmet")
        ? "rgba(16,185,129,0.95)"
        : cls.includes("person")
        ? "rgba(59,130,246,0.9)"
        : "rgba(234,88,12,0.9)";

      // box
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.stroke();

      // label background
      const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
      const textWidth = ctx.measureText(label).width + 8;
      const textHeight = parseInt(ctx.font, 10) + 6;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - textHeight), textWidth, textHeight);

      // label text
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 4, Math.max(0, y - 4));
    });
  }

  const helmetCount = detections.filter((d) => d.class_name.toLowerCase().includes("helmet")).length;
  const personCount = detections.filter((d) => d.class_name.toLowerCase().includes("person")).length;

  // download annotated: render boxes at natural resolution onto offscreen canvas
  const downloadAnnotated = async () => {
    if (!previewUrl) {
      setError("No image to download.");
      return;
    }
    if (!detections.length) {
      setError("No detections to annotate.");
      return;
    }

    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = previewUrl;
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error("Failed to load image for download."));
      });

      const off = document.createElement("canvas");
      off.width = img.naturalWidth;
      off.height = img.naturalHeight;
      const ctx = off.getContext("2d");
      if (!ctx) throw new Error("Failed to create canvas");

      // draw base image
      ctx.drawImage(img, 0, 0, off.width, off.height);

      // draw boxes scaled to natural resolution
      detections.forEach((d) => {
        if (!d.bbox || d.bbox.length < 4) return;
        const [x1, y1, x2, y2] = d.bbox.map(Number);
        const w = x2 - x1;
        const h = y2 - y1;

        const cls = d.class_name.toLowerCase();
        const color = cls.includes("helmet")
          ? "rgba(16,185,129,0.95)"
          : cls.includes("person")
          ? "rgba(59,130,246,0.9)"
          : "rgba(234,88,12,0.9)";

        ctx.lineWidth = Math.max(2, Math.round(off.width / 300));
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.strokeRect(x1, y1, w, h);

        const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
        const fontSize = Math.max(12, Math.round(off.width / 40));
        ctx.font = `${fontSize}px Inter, Arial`;
        const tw = ctx.measureText(label).width + 8;
        const th = fontSize + 6;
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - th), tw, th);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x1 + 4, Math.max(0, y1 - 4));
      });

      off.toBlob((blob) => {
        if (!blob) {
          setError("Failed to create annotated image.");
          return;
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "annotated.png";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }, "image/png");
    } catch (err: any) {
      setError(err?.message || "Failed to download annotated image");
    }
  };

  // Derived booleans for button states
  const canRun = Boolean(file) && !loading;
  const canDownload = detections.length > 0 && !loading;

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-6xl mx-auto">
        <header className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold">🪖 Helmet Detection</h1>
            <p className="text-sm text-slate-500">Upload an image to detect helmets (public API)</p>
          </div>
          <div className="text-sm text-slate-600">API: <code>{API_BASE}</code></div>
        </header>

        <section className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2 bg-white p-5 rounded shadow">
            <div className="flex gap-4">
              <div className="w-1/3">
                {/* Hidden file input */}
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  onChange={onFileChange}
                  style={{ display: "none" }}
                  aria-hidden="true"
                />

                {/* Choose File button */}
                <button
                  onClick={openFilePicker}
                  className="w-full py-2 px-3 mb-2 bg-white border rounded text-slate-700 hover:bg-slate-50"
                  aria-label="Choose file to upload"
                >
                  {file ? `Selected: ${file.name}` : "Choose File"}
                </button>

                {/* Run Detection button */}
                <button
                  onClick={callPredict}
                  disabled={!canRun}
                  aria-disabled={!canRun}
                  aria-busy={loading}
                  className={`w-full py-2 px-3 mb-2 rounded ${canRun ? "bg-indigo-600 text-white hover:bg-indigo-700" : "bg-gray-200 text-gray-500 cursor-not-allowed"}`}
                >
                  {loading ? "Detecting..." : "Run Detection"}
                </button>

                <button
                  onClick={() => {
                    if (previewUrl) URL.revokeObjectURL(previewUrl);
                    setFile(null);
                    setPreviewUrl(null);
                    setDetections([]);
                  }}
                  className="w-full py-2 px-3 border rounded mb-2"
                >
                  Clear
                </button>

                <div className="mt-4 text-sm">
                  <div>Helmets: <strong>{helmetCount}</strong></div>
                  <div>Persons: <strong>{personCount}</strong></div>
                </div>

                <div className="mt-4">
                  <button
                    onClick={downloadAnnotated}
                    disabled={!canDownload}
                    aria-disabled={!canDownload}
                    className={`w-full py-2 px-3 rounded ${canDownload ? "bg-amber-600 text-white hover:bg-amber-700" : "bg-gray-200 text-gray-500 cursor-not-allowed"}`}
                  >
                    Download Annotated
                  </button>
                </div>
              </div>

              <div className="w-2/3 flex items-center justify-center">
                <div style={{ position: "relative" }}>
                  {previewUrl ? (
                    <>
                      <img
                        ref={imgRef}
                        src={previewUrl}
                        alt="preview"
                        className="max-w-full rounded"
                        onLoad={() => drawBoxes()}
                      />
                      <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }} />
                    </>
                  ) : (
                    <div className="text-slate-400">Preview area — upload an image</div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <aside className="bg-white p-5 rounded shadow">
            <h3 className="font-semibold mb-3">Results & Meta</h3>
            {error && <div className="text-sm text-red-600 mb-2">{error}</div>}
            <div className="text-sm text-slate-700 mb-4">
              <div><strong>Model</strong>: helmet_detector_best_20260216_163341.pt</div>
              <div><strong>mAP@0.5</strong>: 0.9045</div>
              <div><strong>mAP@0.5:0.95</strong>: 0.6032</div>
            </div>

            <h4 className="text-sm font-medium mb-2">Detections</h4>
            {detections.length === 0 ? (
              <div className="text-sm text-slate-500">No detections yet</div>
            ) : (
              <ul className="space-y-2 max-h-64 overflow-auto">
                {detections.map((d, i) => (
                  <li key={i} className="p-2 border rounded text-sm">
                    <div className="flex justify-between">
                      <div className="font-medium">{d.class_name}</div>
                      <div className="text-slate-500">{(d.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div className="text-xs text-slate-400 mt-1">BBox: [{d.bbox.map((n) => Math.round(n)).join(", ")}]</div>
                  </li>
                ))}
              </ul>
            )}
          </aside>
        </section>

        <section className="bg-white p-4 mt-6 rounded shadow">
          <h4 className="font-semibold mb-2">Raw API Response</h4>
          <pre className="max-h-48 overflow-auto text-sm bg-slate-50 p-2 rounded">{JSON.stringify({ detections }, null, 2)}</pre>
        </section>
      </div>
    </div>
  );
}
