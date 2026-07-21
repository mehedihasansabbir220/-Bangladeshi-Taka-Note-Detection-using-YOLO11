# 🇧🇩 Bangladeshi Taka Note Detection using YOLO11

A computer vision project that detects and classifies Bangladeshi Taka banknotes in images and video using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics). Built for real-world conditions — varied lighting, angles, and note wear — with a training pipeline anyone can reproduce locally or in Google Colab.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project trains an object detection model to identify Bangladeshi Taka denominations (৳1, ৳2, ৳5, ৳10, ৳20, ৳50, ৳100, ৳500, ৳1000) directly from photos.

> **How detection works here:** the dataset annotates the *printed denomination numerals* on each note (the "৫০০" / "500" marks), not the outline of the note itself. So the model locates and reads denomination markings — which is what actually determines a note's value — rather than drawing a box around the whole banknote.

Potential applications include:

- Assistive tools for visually impaired users
- Automated cash-counting / point-of-sale systems
- Currency-sorting kiosks and vending machines
- Educational computer vision demos

---

## 🎯 Objectives

- Detect Bangladeshi currency notes in real-world images
- Classify each detection into the correct denomination
- Provide a fully reproducible training pipeline (local + Colab)
- Report transparent, honest evaluation metrics

---

## 💰 Supported Classes

| Class ID | Denomination |
|---|---|
| 0 | ৳1 |
| 1 | ৳2 |
| 2 | ৳5 |
| 3 | ৳10 |
| 4 | ৳20 |
| 5 | ৳50 |
| 6 | ৳100 |
| 7 | ৳500 |
| 8 | ৳1000 |

Class IDs are assigned in ascending denomination order by [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py). Note that ৳200 is **not** covered — the source dataset contains no ৳200 images — while ৳1 is.

---

## 📂 Project Structure

```
bangladeshi-taka-detection-yolo11/
├── src/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Validation / metrics script
│   ├── predict.py          # Inference on images/video/webcam
│   └── export.py           # Export best.pt to ONNX/TFLite/NCNN
├── scripts/
│   ├── download_dataset.py # Roboflow dataset downloader
│   └── prepare_dataset.py  # Cleans classes + builds leak-free train/valid/test split
├── docs/
│   └── RESULTS.md          # Evaluation results log (fill in after training)
├── assets/
│   └── sample_predictions/ # Example output images (add your own after training)
├── dataset/
│   ├── train/              # Raw Roboflow download (left untouched)
│   └── prepared/           # Cleaned + split dataset, with generated data.yaml
├── runs/detect/<run-name>/ # Ultralytics training output (weights, plots, logs)
├── dataset.yaml            # Reference copy of the class list / paths
├── best.pt                 # Convenience copy of the latest trained weights (git-ignored)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

This project is built around the **[Bangladeshi Currency Detection dataset](https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection)** by tanvirtain on Roboflow Universe (1,523 images, 11 classes, CC BY 4.0 license).

---

## 🧠 Model Weights

Every training run writes its weights to `runs/detect/<run-name>/weights/best.pt`. For convenience, the most recent run's weights are also kept as `best.pt` in the project root, so every command below can simply use `--weights best.pt`.

To refresh that copy after a new training run:

```bash
cp runs/detect/<run-name>/weights/best.pt ./best.pt
```

> Note: `.pt` files are intentionally **not** committed to this repository (see [`.gitignore`](.gitignore)) — trained weights are large binary files and belong in Releases or a model-hosting service, not version control. After cloning, train your own using the instructions below, or download a released version from the [Releases](../../releases) page once you've published one.

---

## 🛠️ Setup

```bash
git clone https://github.com/mehedihasansabbir220/-Bangladeshi-Taka-Note-Detection-using-YOLO11.git
cd bangladeshi-taka-detection-yolo11

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Configure your Roboflow API key (kept out of git via .gitignore)
cp .env.example .env
# Edit .env and set ROBOFLOW_API_KEY=...
```

---

## 📥 Get the Dataset

```bash
# Reads ROBOFLOW_API_KEY from .env
python3 scripts/download_dataset.py

# Or pass the key explicitly:
python3 scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY
```

This downloads the dataset into `./dataset/` in YOLO format.

---

## 🧹 Prepare the Dataset

**Do not train on the raw download.** The Roboflow export ships three problems that make honest evaluation impossible:

| Problem | Consequence |
|---|---|
| `val` points at the training images | Any reported mAP is the model grading its own homework |
| 2–3 augmented copies of each of the 1,166 source photos | A naive random split scatters copies of one photo across train *and* val, leaking the answer |
| `500 taka` and `Five Hundred taka` are duplicate classes; `currency` boxes whole notes while every other class boxes numerals | The model learns to confuse annotation concepts |

One command fixes all three:

```bash
python3 scripts/prepare_dataset.py
```

It merges the duplicate ৳500 classes, drops the 212 inconsistent `currency` boxes, renumbers the remaining 9 classes in denomination order, and — critically — splits **by source photo**, so every augmented copy of a photo lands in the same split. Output goes to `dataset/prepared/` with a generated `data.yaml`; the raw download is left untouched.

Resulting split (70/15/15 by photo, seed 42):

| Split | Photos | Images | Boxes |
|---|---|---|---|
| train | 816 | 2,409 | 6,036 |
| valid | 175 | 512 | 1,325 |
| test | 175 | 519 | 1,288 |

---

## 🚀 Train

```bash
python3 src/train.py --data dataset/prepared/data.yaml --model yolo11s.pt --epochs 100
```

Key flags:
| Flag | Default | Description |
|---|---|---|
| `--model` | `yolo11s.pt` | Base checkpoint: `yolo11n/s/m/l/x.pt` |
| `--epochs` | `100` | Training epochs |
| `--imgsz` | `640` | Input image size |
| `--batch` | `16` | Batch size |
| `--device` | auto-detected | `0` for first GPU, `cpu` to force CPU |

Trained weights are saved to `runs/detect/<run-name>/weights/best.pt` (see [Model Weights](#-model-weights)).

---

## 📈 Evaluate

```bash
# Tune against the validation split while iterating
python3 src/evaluate.py --weights best.pt --data dataset/prepared/data.yaml --split val

# Report your final number on the held-out test split — once, at the end
python3 src/evaluate.py --weights best.pt --data dataset/prepared/data.yaml --split test
```

Prints overall and per-class mAP50 / mAP50-95, precision, and recall. Log your results in [`docs/RESULTS.md`](docs/RESULTS.md) so the project has a transparent record.

Read the **per-class** numbers, not just the average — a healthy-looking mAP often hides one denomination sitting at 0.2. ৳500 is the one to watch: it has the fewest boxes (252 in train) because its source photos were annotated differently from the rest. Also check `runs/detect/val/confusion_matrix.png` to see *which* denominations get mistaken for each other.

---

## 🔍 Run Inference

```bash
# On a folder of images
python3 src/predict.py --weights best.pt --source path/to/images

# On a single image
python3 src/predict.py --weights best.pt --source path/to/note.jpg

# On a webcam
python3 src/predict.py --weights best.pt --source 0
```

---

## 📦 Export for Deployment

```bash
python3 src/export.py --weights best.pt --format onnx
```

Supported formats: `onnx`, `tflite`, `ncnn`, `torchscript`.

---

## 📊 Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for the current evaluation log. This section is intentionally left for you to fill in with your own training run — reporting fabricated metrics before you've actually trained the model would misrepresent the project.

---

## 🗺️ Roadmap

- [x] Build a leak-free train/valid/test split and clean up duplicate classes
- [ ] Train baseline `yolo11s` model and log results
- [ ] Expand dataset with additional real-world photos (varied lighting/angle)
- [ ] Address class imbalance on ৳500 (fewest boxes) and ৳1000
- [ ] Add ৳200 coverage — the denomination is missing from the source dataset entirely
- [ ] Export to TFLite for a mobile demo app
- [ ] Add a simple Streamlit/Gradio demo UI

---

## 🤝 Contributing

Issues and pull requests are welcome — especially additional labeled images of underrepresented denominations.

---

## 📄 License

This project is released under the [MIT License](LICENSE). The dataset used is licensed CC BY 4.0 by its original author (tanvirtain, Roboflow Universe) — please retain attribution if you redistribute it.

---

## 🙏 Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [Bangladeshi Currency Detection dataset](https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection) by tanvirtain
