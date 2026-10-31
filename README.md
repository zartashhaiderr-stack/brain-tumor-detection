# Brain Tumor Detection from MRI (Tumor vs No Tumor)

A deep learning model that looks at a brain MRI scan and says whether a tumor is present.
EfficientNetB0 (pretrained on ImageNet, then fine-tuned) is compared against a custom CNN and a frozen VGG16 baseline.
The final model runs in the browser on the live website.

**GitHub repository:** https://github.com/zartashhaiderr-stack/brain-tumor-detection
**Dataset (Kaggle):** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Results (test set)
Full table: [`results/metrics.md`](results/metrics.md). Charts are in [`results/`](results/).

## Quick test for evaluators
```bash
pip install -r requirements.txt
python predict.py path/to/image.jpg            # one scan
python predict.py path/to/folder               # every image in a folder
python predict.py path/to/Testing --eval       # folders like glioma/ notumor/ ... gives accuracy, sensitivity, specificity
python predict.py path/to/image.jpg --json     # machine-readable output
```

| Item | Location |
|---|---|
| Model weights (Keras) | `weights/brain_tumor_effnetb0.keras` |
| Model weights (ONNX) | `weights/brain_tumor_effnetb0.onnx` |
| Model settings (input size, threshold, labels) | `weights/config.json` |
| Training notebook | `notebooks/brain_tumor_detection_colab.ipynb` |
| Prediction script | `predict.py` |
| Website (Vercel) | `web/` |

**Input format:** RGB image resized to 224 x 224, raw pixel values 0 to 255, shape `[N, 224, 224, 3]`, float32. No normalization needed (the model does it internally).
**Output:** `tumor_probability` from 0 to 1. Scan is classed as Tumor when it is at or above the threshold in `weights/config.json`.

## Dataset
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (secondary dataset).
Glioma, meningioma and pituitary are grouped as **Tumor**; notumor is **No tumor**.
The notebook downloads it automatically. Images are not stored in this repo; a few test scans are in `web/samples/`.

## Method
1. Removed duplicate images (MD5 hash) so no scan is in both training and testing.
2. Split: 85% train / 15% validation from the Training folder; the Testing folder is used only once for the final score.
3. Augmentation: flips, small rotations, zoom, shifts, contrast.
4. Class weights to handle the 3 to 1 tumor vs no tumor imbalance.
5. EfficientNetB0: train the new head with the base frozen, then fine-tune the whole network at a low learning rate (BatchNorm frozen).
6. Decision threshold picked on the validation set to catch at least 98% of tumors.
7. Grad-CAM heatmaps to check the model looks at the right areas.
8. Exported to ONNX and checked that Keras and ONNX give the same outputs.

## Repo structure
```
├── notebooks/brain_tumor_detection_colab.ipynb   training (Google Colab)
├── predict.py                                    command-line prediction
├── requirements.txt
├── weights/                                      model files + config.json
├── results/                                      charts and metric tables
└── web/                                          Vercel website (runs the model in the browser)
```

## Disclaimer
Student project for learning purposes. Not a medical device and not for diagnosis.
