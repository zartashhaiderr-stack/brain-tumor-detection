| Model | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | Missed tumors | False alarms |
|---|---|---|---|---|---|---|---|---|---|
| Custom CNN | 0.1674 | 92.23% | 89.70% | 99.75% | 99.91% | 94.53% | 0.9823 | 122 | 1 |
| VGG16 (frozen) | 0.5 | 96.40% | 95.52% | 99.00% | 99.65% | 97.54% | 0.9943 | 53 | 4 |
| EfficientNetB0 (fine-tuned) | 0.5 | 97.35% | 96.45% | 100.00% | 100.00% | 98.19% | 0.9980 | 42 | 0 |
