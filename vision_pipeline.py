from __future__ import annotations

import csv
import json
import os
import random
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


@dataclass
class TrainConfig:
    dataset_path: str
    output_dir: str = "runs_custom"
    base_model: str = "yolo11n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    class_names_csv: str = ""
    val_ratio: float = 0.2
    seed: int = 42


@dataclass
class ValidateConfig:
    model_path: str
    video_path: str
    output_dir: str = "runs_custom/validation"
    conf_threshold: float = 0.25
    display_threshold: float = 0.40
    save_threshold: float = 0.50
    frame_skip: int = 0
    line_width: int = 2
    font_scale: float = 0.7
    show_window: bool = True
    save_video: bool = True
    use_tracking: bool = True
    tracker_yaml: str = "bytetrack.yaml"


class PipelineError(Exception):
    pass


class StatusReporter:
    def __call__(self, message: str) -> None:
        print(message)


class VisionPipeline:
    def __init__(self, reporter: Optional[StatusReporter] = None):
        self.report = reporter or StatusReporter()

    # -------------------------
    # General helpers
    # -------------------------
    def _mkdir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _parse_class_names(self, csv_text: str) -> Optional[List[str]]:
        names = [x.strip() for x in csv_text.split(",") if x.strip()]
        return names or None

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTS

    def _color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        # Deterministic BGR colors.
        palette = [
            (56, 56, 255),
            (151, 157, 255),
            (31, 112, 255),
            (29, 178, 255),
            (49, 210, 207),
            (10, 249, 72),
            (23, 204, 146),
            (134, 219, 61),
            (52, 147, 26),
            (187, 212, 0),
            (168, 153, 44),
            (255, 194, 0),
            (147, 69, 52),
            (255, 115, 100),
            (236, 24, 0),
            (255, 56, 132),
            (133, 0, 82),
            (255, 56, 203),
            (200, 149, 255),
            (199, 55, 255),
        ]
        return palette[class_id % len(palette)]

    def _format_timestamp(self, frame_idx: int, fps: float) -> str:
        seconds = frame_idx / fps if fps > 0 else 0.0
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

    def _extract_if_zip(self, input_path: Path, work_dir: Path) -> Path:
        if input_path.suffix.lower() != ".zip":
            return input_path
        extract_dir = self._mkdir(work_dir / "extracted_dataset")
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
        children = [p for p in extract_dir.iterdir()]
        return children[0] if len(children) == 1 and children[0].is_dir() else extract_dir

    def _find_dataset_yaml(self, root: Path) -> Optional[Path]:
        candidates = list(root.rglob("dataset.yaml")) + list(root.rglob("data.yaml"))
        return candidates[0] if candidates else None

    def _find_coco_json(self, root: Path) -> Optional[Path]:
        candidates = [
            p for p in root.rglob("*.json")
            if p.name.lower() in {"instances_train.json", "instances_val.json", "annotations.json", "coco.json"}
        ]
        return candidates[0] if candidates else None

    def _load_yaml_names(self, yaml_path: Path) -> List[str]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, list):
            return names
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=lambda x: int(x))]
        return []

    # -------------------------
    # Dataset preparation
    # -------------------------
    def _find_yolo_layout(self, base: Path) -> Optional[Dict[str, Path]]:
        layouts = [
            {
                "train_images": base / "images" / "train",
                "val_images": base / "images" / "val",
                "train_labels": base / "labels" / "train",
                "val_labels": base / "labels" / "val",
            },
            {
                "train_images": base / "train" / "images",
                "val_images": base / "val" / "images",
                "train_labels": base / "train" / "labels",
                "val_labels": base / "val" / "labels",
            },
        ]
        for layout in layouts:
            if layout["train_images"].exists() and layout["train_labels"].exists():
                return layout
        return None

    def _infer_class_count_from_labels(self, label_dirs: List[Path]) -> int:
        max_cls = -1
        for label_dir in label_dirs:
            if not label_dir.exists():
                continue
            for txt in label_dir.rglob("*.txt"):
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                max_cls = max(max_cls, int(float(parts[0])))
                except Exception:
                    continue
        return max_cls + 1 if max_cls >= 0 else 0

    def _split_train_into_val(
        self,
        train_images_dir: Path,
        train_labels_dir: Path,
        out_base: Path,
        val_ratio: float,
        seed: int,
    ) -> Path:
        random.seed(seed)
        all_images = sorted([p for p in train_images_dir.rglob("*") if p.is_file() and self._is_image(p)])
        pairs = []
        for img in all_images:
            rel = img.relative_to(train_images_dir)
            label = (train_labels_dir / rel).with_suffix(".txt")
            if label.exists():
                pairs.append((img, label))
        if len(pairs) < 2:
            raise PipelineError("Not enough labeled train samples to create a validation split.")
        random.shuffle(pairs)
        split_idx = max(1, int(len(pairs) * (1 - val_ratio)))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        new_root = self._mkdir(out_base / "prepared_dataset")
        ti = self._mkdir(new_root / "images" / "train")
        vi = self._mkdir(new_root / "images" / "val")
        tl = self._mkdir(new_root / "labels" / "train")
        vl = self._mkdir(new_root / "labels" / "val")
        for img, lbl in train_pairs:
            shutil.copy2(img, ti / img.name)
            shutil.copy2(lbl, tl / lbl.name)
        for img, lbl in val_pairs:
            shutil.copy2(img, vi / img.name)
            shutil.copy2(lbl, vl / lbl.name)
        return new_root

    def _write_dataset_yaml(self, root: Path, names: List[str]) -> Path:
        yaml_path = root / "dataset.yaml"
        data = {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(names)},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return yaml_path

    def _convert_single_coco_json(self, json_path: Path, output_root: Path, val_ratio: float, seed: int) -> Path:
        self.report(f"Converting COCO annotations: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco.get("images", [])}
        categories = sorted(coco.get("categories", []), key=lambda x: x["id"])
        cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
        names = [cat["name"] for cat in categories]

        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        source_root = json_path.parent
        pairs = []
        for img_id, img_info in images.items():
            file_name = img_info.get("file_name")
            if not file_name:
                continue
            img_path = source_root / file_name
            if not img_path.exists():
                matches = list(source_root.rglob(Path(file_name).name))
                if not matches:
                    continue
                img_path = matches[0]
            pairs.append((img_id, img_path, img_info))

        if not pairs:
            raise PipelineError("COCO JSON found, but no referenced images could be resolved.")

        random.seed(seed)
        random.shuffle(pairs)
        split_idx = max(1, int(len(pairs) * (1 - val_ratio)))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        root = self._mkdir(output_root / "coco_converted")
        train_img_dir = self._mkdir(root / "images" / "train")
        val_img_dir = self._mkdir(root / "images" / "val")
        train_lbl_dir = self._mkdir(root / "labels" / "train")
        val_lbl_dir = self._mkdir(root / "labels" / "val")

        def save_label_file(dst_img_dir: Path, dst_lbl_dir: Path, item: Tuple[int, Path, dict]) -> None:
            img_id, src_img, img_info = item
            dst_img = dst_img_dir / src_img.name
            shutil.copy2(src_img, dst_img)
            width = float(img_info["width"])
            height = float(img_info["height"])
            label_path = dst_lbl_dir / f"{src_img.stem}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                for ann in anns_by_img.get(img_id, []):
                    if "bbox" not in ann:
                        continue
                    x, y, w, h = ann["bbox"]
                    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
                        continue
                    xc = (x + w / 2) / width
                    yc = (y + h / 2) / height
                    wn = w / width
                    hn = h / height
                    cls_idx = cat_id_to_idx[ann["category_id"]]
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        for item in train_pairs:
            save_label_file(train_img_dir, train_lbl_dir, item)
        for item in val_pairs:
            save_label_file(val_img_dir, val_lbl_dir, item)

        return self._write_dataset_yaml(root, names)

    def prepare_dataset(self, cfg: TrainConfig) -> Tuple[Path, Path]:
        dataset_input = Path(cfg.dataset_path)
        if not dataset_input.exists():
            raise PipelineError(f"Dataset path does not exist: {dataset_input}")

        output_dir = self._mkdir(Path(cfg.output_dir))
        work_dir = self._mkdir(output_dir / "work")
        dataset_root = self._extract_if_zip(dataset_input, work_dir)

        yaml_path = self._find_dataset_yaml(dataset_root)
        if yaml_path:
            self.report(f"Using existing dataset YAML: {yaml_path}")
            return yaml_path, dataset_root

        coco_json = self._find_coco_json(dataset_root)
        if coco_json:
            yaml_path = self._convert_single_coco_json(coco_json, work_dir, cfg.val_ratio, cfg.seed)
            self.report(f"COCO dataset converted to YOLO format: {yaml_path}")
            return yaml_path, yaml_path.parent

        layout = self._find_yolo_layout(dataset_root)
        if not layout:
            raise PipelineError(
                "Unsupported dataset structure. Provide either a dataset.yaml, a COCO JSON export, or a YOLO layout."
            )

        train_images = layout["train_images"]
        val_images = layout["val_images"]
        train_labels = layout["train_labels"]
        val_labels = layout["val_labels"]

        root_for_yaml = dataset_root
        if not val_images.exists() or not val_labels.exists():
            self.report("Validation split not found. Creating one automatically.")
            root_for_yaml = self._split_train_into_val(train_images, train_labels, work_dir, cfg.val_ratio, cfg.seed)
            layout = self._find_yolo_layout(root_for_yaml)
            assert layout is not None
            train_labels = layout["train_labels"]
            val_labels = layout["val_labels"]

        class_names = self._parse_class_names(cfg.class_names_csv)
        class_count = self._infer_class_count_from_labels([train_labels, val_labels])
        if class_count <= 0:
            raise PipelineError("Could not infer class count from YOLO label files.")
        if class_names is None:
            class_names = [f"class_{i}" for i in range(class_count)]
        if len(class_names) != class_count:
            raise PipelineError(
                f"Class name count ({len(class_names)}) does not match inferred class count ({class_count})."
            )
        yaml_path = self._write_dataset_yaml(root_for_yaml, class_names)
        self.report(f"Generated dataset YAML: {yaml_path}")
        return yaml_path, root_for_yaml

    # -------------------------
    # Training / export
    # -------------------------
    def train(self, cfg: TrainConfig) -> Path:
        dataset_yaml, _ = self.prepare_dataset(cfg)
        self.report("Starting training...")
        model = YOLO(cfg.base_model)
        results = model.train(
            data=str(dataset_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            project=str(Path(cfg.output_dir) / "training"),
            name="custom_detector",
            pretrained=True,
            seed=cfg.seed,
        )
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if not best_model_path.exists():
            raise PipelineError(f"Training completed but best.pt was not found: {best_model_path}")
        self.report(f"Training complete. Best model: {best_model_path}")
        return best_model_path

    def export_model(self, model_path: str, export_format: str, output_dir: str) -> Optional[str]:
        export_format = export_format.strip().lower()
        if export_format not in {"none", "onnx", "engine"}:
            raise PipelineError("Export format must be one of: none, onnx, engine")
        if export_format == "none":
            return None
        self.report(f"Exporting model as {export_format}...")
        model = YOLO(model_path)
        exported = model.export(format=export_format, project=output_dir, name=f"export_{export_format}")
        self.report(f"Export completed: {exported}")
        return str(exported)

    # -------------------------
    # Validation / tracking
    # -------------------------
    def validate_video(self, cfg: ValidateConfig) -> Dict[str, str]:
        model_path = Path(cfg.model_path)
        video_path = Path(cfg.video_path)
        if not model_path.exists():
            raise PipelineError(f"Model path does not exist: {model_path}")
        if not video_path.exists() or video_path.suffix.lower() not in VIDEO_EXTS:
            raise PipelineError(f"Video path is invalid or unsupported: {video_path}")

        out_dir = self._mkdir(Path(cfg.output_dir))
        model = YOLO(str(model_path))
        names = model.names

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PipelineError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        writer = None
        out_video_path = out_dir / f"{video_path.stem}_annotated.mp4"
        if cfg.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        csv_path = out_dir / f"{video_path.stem}_detections.csv"
        json_path = out_dir / f"{video_path.stem}_detections.json"
        summary_path = out_dir / f"{video_path.stem}_summary.csv"

        all_rows: List[dict] = []
        frame_idx = 0
        processed_frames = 0
        frame_skip = max(0, int(cfg.frame_skip))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "frame_idx", "timestamp", "label", "class_id", "track_id", "confidence",
                "x1", "y1", "x2", "y2"
            ])

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = self._format_timestamp(frame_idx, fps)
                should_process = (frame_idx % (frame_skip + 1) == 0)
                annotated = frame.copy()

                detections_this_frame = []
                if should_process:
                    processed_frames += 1
                    if cfg.use_tracking:
                        results = model.track(
                            source=frame,
                            conf=cfg.conf_threshold,
                            tracker=cfg.tracker_yaml,
                            persist=True,
                            verbose=False,
                        )
                    else:
                        results = model.predict(
                            source=frame,
                            conf=cfg.conf_threshold,
                            verbose=False,
                        )

                    if results:
                        result = results[0]
                        boxes = result.boxes
                        if boxes is not None:
                            ids = None
                            try:
                                if boxes.id is not None:
                                    ids = boxes.id.int().cpu().tolist()
                            except Exception:
                                ids = None

                            for i in range(len(boxes)):
                                box = boxes[i]
                                cls_id = int(box.cls.item())
                                conf = float(box.conf.item())
                                if conf < cfg.save_threshold:
                                    continue
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                                track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
                                detections_this_frame.append({
                                    "frame_idx": frame_idx,
                                    "timestamp": timestamp,
                                    "label": label,
                                    "class_id": cls_id,
                                    "track_id": track_id,
                                    "confidence": conf,
                                    "bbox_xyxy": [x1, y1, x2, y2],
                                })

                for det in detections_this_frame:
                    cls_id = det["class_id"]
                    conf = det["confidence"]
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    track_id = det["track_id"]
                    color = self._color_for_class(cls_id)

                    if conf >= cfg.display_threshold:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, cfg.line_width)
                        id_text = f" ID:{track_id}" if track_id >= 0 else ""
                        caption = f"{det['label']}{id_text} {conf:.2f}"
                        cv2.putText(
                            annotated,
                            caption,
                            (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            cfg.font_scale,
                            color,
                            2,
                        )

                    writer_csv.writerow([
                        det["frame_idx"], det["timestamp"], det["label"], det["class_id"], det["track_id"],
                        f"{det['confidence']:.6f}", x1, y1, x2, y2
                    ])
                    all_rows.append(det)

                overlay_lines = [
                    f"Frame: {frame_idx}",
                    f"Time: {timestamp}",
                    f"Processed every {frame_skip + 1} frame(s)",
                    f"Tracking: {'ON' if cfg.use_tracking else 'OFF'}",
                    f"Display>{cfg.display_threshold:.2f} Save>{cfg.save_threshold:.2f}",
                ]
                for idx, line in enumerate(overlay_lines):
                    cv2.putText(
                        annotated,
                        line,
                        (20, 30 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                if writer is not None:
                    writer.write(annotated)
                if cfg.show_window:
                    cv2.imshow("Validation - detections", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        break

                frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)

        summary_rows = {}
        for row in all_rows:
            label = row["label"]
            summary_rows.setdefault(label, {"detections": 0, "confidence_sum": 0.0, "track_ids": set()})
            summary_rows[label]["detections"] += 1
            summary_rows[label]["confidence_sum"] += row["confidence"]
            if row["track_id"] >= 0:
                summary_rows[label]["track_ids"].add(row["track_id"])

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["label", "detections", "unique_tracks", "avg_confidence"])
            for label, stats in sorted(summary_rows.items(), key=lambda x: x[1]["detections"], reverse=True):
                avg_conf = stats["confidence_sum"] / max(1, stats["detections"])
                writer_csv.writerow([label, stats["detections"], len(stats["track_ids"]), f"{avg_conf:.6f}"])

        self.report(f"Validation complete. Processed frames: {processed_frames}")
        return {
            "csv": str(csv_path),
            "json": str(json_path),
            "summary": str(summary_path),
            "video": str(out_video_path) if cfg.save_video else "",
        }
