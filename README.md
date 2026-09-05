# Brain-Tumor-Detection

A YOLOv8-based brain tumor detection project built with Streamlit. The app lets you upload an MRI image and view the detected tumor classes, confidence scores, and bounding boxes.

## Features

- Upload an MRI slice image in the browser
- Run inference with a pretrained YOLOv8 model
- View annotated predictions and detection details
- Adjust confidence and IoU thresholds from the sidebar

## Project Files

- `app.py` - Streamlit web app
- `yolov8_brain_tumor_best.pt` - main trained model weights
- `yolov8s.pt` - fallback YOLOv8 weights
- `dataset/` - training, validation, and test dataset
- `data.yaml` / `dataset.yaml` - dataset configuration files

## Requirements

This project uses Python 3.13+ and the dependencies listed in `pyproject.toml`.

## Run the App

Using `uv`:

```bash
uv sync
uv run streamlit run app.py
```

If you prefer to manage packages another way, install the dependencies listed in `pyproject.toml`, then run:

```bash
streamlit run app.py
```

## Notes

- The model is intended for research and educational use only.
- The app will automatically try to load `yolov8_brain_tumor_best.pt` first, then fall back to `yolov8s.pt` if needed.
