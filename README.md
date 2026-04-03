# Custom Vision Trainer + Video Validator

This project gives you a **two-file Python app**:

- `vision_pipeline.py` — reusable training / export / validation logic
- `app_gui.py` — desktop GUI for selecting a dataset, training a model, exporting it, and validating it on a video

## Included upgrades

- object tracking with persistent IDs across frames
- per-class box colors
- COCO JSON import and conversion to YOLO format
- GUI with Train and Validate tabs
- export to ONNX or TensorRT engine after training
- separate thresholds for inference, display, and saving results
- frame skipping for faster validation
- timestamps, labels, confidence, track IDs, and boxes saved to CSV and JSON
- annotated output video

## Install

```bash
pip install ultralytics opencv-python pyyaml
```

Tkinter is included with most Python desktop installs. If it is missing on Linux, install your distro's tkinter package.

## Run the GUI

```bash
python app_gui.py
```

## Dataset formats supported

### 1. YOLO dataset with dataset.yaml

```text
my_dataset/
  dataset.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

### 2. YOLO dataset without dataset.yaml

```text
my_dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

Or the equivalent `images/train`, `labels/train`, `images/val`, `labels/val` layout.

### 3. COCO JSON

The app will look for a COCO-style annotation JSON and convert it into YOLO format automatically.

## Notes

- Tracking uses Ultralytics tracking with `bytetrack.yaml` by default.
- TensorRT export (`engine`) typically requires a proper TensorRT-enabled environment.
- The validator saves only detections above the **save threshold**, and only draws boxes above the **display threshold**.
- Frame skip `0` means every frame is processed. Frame skip `2` means process every 3rd frame.

## Output files

The validation step writes:

- `*_detections.csv`
- `*_detections.json`
- `*_summary.csv`
- `*_annotated.mp4`
