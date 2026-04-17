from __future__ import annotations

CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_COL = "Image Index"
LABEL_COL = "Finding Labels"
PATIENT_COL = "Patient ID"

DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = [round(x, 2) for x in [0.1 + 0.05 * i for i in range(17)]]
