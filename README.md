# 🇧🇩 Bangladeshi Taka Note Detection using YOLO11

A computer vision project that detects and classifies Bangladeshi Taka banknotes in images and video using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics). Built for real-world conditions — varied lighting, angles, and note wear — with a training pipeline anyone can reproduce locally or in Google Colab.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project trains an object detection model to identify Bangladeshi Taka denominations (৳2, ৳5, ৳10, ৳20, ৳50, ৳100, ৳200, ৳500, ৳1000) directly from photos. Potential applications include:

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
| 0 | ৳2 |
| 1 | ৳5 |
| 2 | ৳10 |
| 3 | ৳20 |
| 4 | ৳50 |
| 5 | ৳100 |
| 6 | ৳200 |
| 7 | ৳500 |
| 8 | ৳1000 |

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
│   └── download_dataset.py # Roboflow dataset downloader
├── docs/
│   └── RESULTS.md          # Evaluation results log (fill in after training)
├── assets/
│   └── sample_predictions/ # Example output images (add your own after training)
├── dataset.yaml            # Dataset config for Ultralytics
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

This project is built around the **[Bangladeshi Currency Detection dataset](https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection)** by tanvirtain on Roboflow Universe (1,523 images, 11 classes, CC BY 4.0 license).

> Note: `best.pt` is intentionally **not** committed to this repository (see [`.gitignore`](.gitignore)) — trained weights are large binary files and belong in Releases or a model-hosting service, not version control. Train your own using the instructions below, or download a released version from the [Releases](../../releases) page once you've published one.

---

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/bangladeshi-taka-detection-yolo11.git
cd bangladeshi-taka-detection-yolo11

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 📥 Get the Dataset

```bash
python3 scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY
```

This downloads the dataset into `./dataset/` in YOLO format, ready to train against.

---

## 🚀 Train

```bash
python3 src/train.py --data dataset/data.yaml --model yolo11s.pt --epochs 100
```

Key flags:
| Flag | Default | Description |
|---|---|---|
| `--model` | `yolo11s.pt` | Base checkpoint: `yolo11n/s/m/l/x.pt` |
| `--epochs` | `100` | Training epochs |
| `--imgsz` | `640` | Input image size |
| `--batch` | `16` | Batch size |
| `--device` | auto-detected | `0` for first GPU, `cpu` to force CPU |

Trained weights are saved to `runs/detect/<run-name>/weights/best.pt`.

---

## 📈 Evaluate

```bash
python3 src/evaluate.py --weights runs/detect/train/weights/best.pt --data dataset/data.yaml
```

Prints overall and per-class mAP50 / mAP50-95, precision, and recall. Log your results in [`docs/RESULTS.md`](docs/RESULTS.md) so the project has a transparent record.

---

## 🔍 Run Inference

```bash
# On a folder of images
python3 src/predict.py --weights runs/detect/train/weights/best.pt --source path/to/images

# On a webcam
python3 src/predict.py --weights runs/detect/train/weights/best.pt --source 0
```

---

## 📦 Export for Deployment

```bash
python3 src/export.py --weights runs/detect/train/weights/best.pt --format onnx
```

Supported formats: `onnx`, `tflite`, `ncnn`, `torchscript`.

---

## 📊 Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for the current evaluation log. This section is intentionally left for you to fill in with your own training run — reporting fabricated metrics before you've actually trained the model would misrepresent the project.

---

## 🗺️ Roadmap

- [ ] Train baseline `yolo11s` model and log results
- [ ] Expand dataset with additional real-world photos (varied lighting/angle)
- [ ] Address class imbalance on higher denominations (৳500, ৳1000)
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
