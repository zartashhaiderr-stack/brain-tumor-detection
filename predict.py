"""
Brain tumor detection from MRI: command-line predictor.

Examples
  python predict.py scan.jpg                       # one image
  python predict.py folder/                        # every image in a folder
  python predict.py Testing/ --eval                # folder with class subfolders -> accuracy report
  python predict.py scan.jpg --json                # machine-readable output
  python predict.py scan.jpg --model weights/brain_tumor_effnetb0.keras   # use the Keras file instead

By default it uses the ONNX file, which needs only: numpy, pillow, onnxruntime.
Settings (image size, threshold, labels) are read from weights/config.json.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "weights", "brain_tumor_effnetb0.onnx")
DEFAULT_CONFIG = os.path.join(HERE, "weights", "config.json")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
NO_TUMOR_NAMES = {"notumor", "no", "healthy", "normal"}


def load_config(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"input": {"shape": [None, 224, 224, 3]}, "threshold": 0.5,
            "labels": {"0": "No tumor", "1": "Tumor"}}


def preprocess(path, size):
    """Exactly what training did: RGB, resize to size x size, raw 0-255 pixels."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


class Model:
    def __init__(self, model_path):
        self.kind = "onnx" if model_path.endswith(".onnx") else "keras"
        if self.kind == "onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
        else:
            import keras
            self.model = keras.models.load_model(model_path, compile=False)

    def __call__(self, batch):
        if self.kind == "onnx":
            return self.sess.run(None, {self.input_name: batch})[0].ravel()
        return self.model.predict(batch, verbose=0).ravel()


def list_images(target):
    if os.path.isfile(target):
        return [target]
    found = []
    for root, _, files in os.walk(target):
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXT):
                found.append(os.path.join(root, f))
    return sorted(found)


def true_label_from_path(path):
    folder = os.path.basename(os.path.dirname(path)).lower().replace("_", "").replace(" ", "").replace("-", "")
    if folder in NO_TUMOR_NAMES:
        return 0
    if folder in {"tumor", "yes", "glioma", "meningioma", "pituitary"}:
        return 1
    return None


def main():
    ap = argparse.ArgumentParser(description="Predict tumor / no tumor for brain MRI images.")
    ap.add_argument("target", help="an image file or a folder of images")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=".onnx (default) or .keras file")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--threshold", type=float, help="override the decision threshold from config.json")
    ap.add_argument("--eval", action="store_true", help="read true labels from folder names and report accuracy")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    ap.add_argument("--csv", help="also save results to this CSV file")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = int(cfg["input"]["shape"][1])
    threshold = args.threshold if args.threshold is not None else float(cfg.get("threshold", 0.5))
    labels = cfg.get("labels", {"0": "No tumor", "1": "Tumor"})

    if not os.path.exists(args.model):
        sys.exit(f"Model not found: {args.model}")
    paths = list_images(args.target)
    if not paths:
        sys.exit(f"No images found at {args.target}")

    model = Model(args.model)
    results = []
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        probs = model(np.stack([preprocess(p, size) for p in chunk]))
        for p, prob in zip(chunk, probs):
            pred = int(prob >= threshold)
            results.append({"file": p, "prediction": labels[str(pred)], "label": pred,
                            "tumor_probability": round(float(prob), 6), "threshold": threshold})

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        for r in results:
            print(f"{r['prediction']:9s}  p(tumor)={r['tumor_probability']:.4f}  {r['file']}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    if args.eval:
        pairs = [(true_label_from_path(r["file"]), r["label"]) for r in results]
        pairs = [(t, p) for t, p in pairs if t is not None]
        if not pairs:
            sys.exit("--eval needs class folders such as notumor/, glioma/, tumor/ ...")
        t = np.array([a for a, _ in pairs]); p = np.array([b for _, b in pairs])
        tp = int(((t == 1) & (p == 1)).sum()); tn = int(((t == 0) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
        report = {"images": len(t), "accuracy": round((tp + tn) / len(t), 4),
                  "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
                  "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
                  "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": threshold}
        print("\nEVALUATION")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
