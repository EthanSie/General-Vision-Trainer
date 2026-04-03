from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from vision_pipeline import PipelineError, TrainConfig, ValidateConfig, VisionPipeline


YOLO_MODEL_OPTIONS = [
    ("yolo11n.pt", "nano (fastest, lowest VRAM, good for CPU/basic GPU)"),
    ("yolo11s.pt", "small (best general starting point, solid mid-range GPU)"),
    ("yolo11m.pt", "medium (better accuracy, needs a solid GPU)"),
    ("yolo11l.pt", "large (high accuracy, strong GPU recommended)"),
    ("yolo11x.pt", "xlarge (highest accuracy, heavy VRAM use, high-end GPU)"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Custom Vision Trainer + Validator")
        self.geometry("980x860")
        self.minsize(920, 780)

        self.pipeline = VisionPipeline(reporter=self.log)
        self.best_model_path_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        train_tab = ttk.Frame(notebook)
        validate_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="Train")
        notebook.add(validate_tab, text="Validate")

        self._build_train_tab(train_tab)
        self._build_validate_tab(validate_tab)
        self._build_log_panel()

    def _build_train_tab(self, parent):
        frame = ttk.Frame(parent, padding=12)
        frame.pack(fill="both", expand=True)

        self.dataset_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path("runs_custom").resolve()))
        self.base_model_var = tk.StringVar(value=YOLO_MODEL_OPTIONS[0][0])
        self.base_model_display_var = tk.StringVar(value=self._format_model_option(YOLO_MODEL_OPTIONS[0]))
        self.epochs_var = tk.StringVar(value="50")
        self.imgsz_var = tk.StringVar(value="640")
        self.batch_var = tk.StringVar(value="16")
        self.class_names_var = tk.StringVar(value="")
        self.val_ratio_var = tk.StringVar(value="0.2")
        self.seed_var = tk.StringVar(value="42")
        self.export_format_var = tk.StringVar(value="none")

        row = 0
        self._path_row(frame, row, "Dataset folder or ZIP", self.dataset_var, self._pick_dataset); row += 1
        self._path_row(frame, row, "Output directory", self.output_var, self._pick_output_dir, directory=True); row += 1
        self._base_model_row(frame, row); row += 1
        self._entry_row(frame, row, "Epochs", self.epochs_var); row += 1
        self._entry_row(frame, row, "Image size", self.imgsz_var); row += 1
        self._entry_row(frame, row, "Batch size", self.batch_var); row += 1
        self._entry_row(frame, row, "Class names (optional CSV)", self.class_names_var); row += 1
        self._entry_row(frame, row, "Validation split ratio", self.val_ratio_var); row += 1
        self._entry_row(frame, row, "Random seed", self.seed_var); row += 1

        ttk.Label(frame, text="Export after training").grid(row=row, column=0, sticky="w", pady=6)
        export_combo = ttk.Combobox(frame, textvariable=self.export_format_var, state="readonly", values=["none", "onnx", "engine"])
        export_combo.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=6)
        row += 1

        ttk.Label(frame, text="Best trained model").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.best_model_path_var).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=6)
        row += 1

        ttk.Button(frame, text="Start Training", command=self.start_training).grid(row=row, column=0, pady=12, sticky="w")
        ttk.Button(frame, text="Copy model path to Validate tab", command=self.copy_model_to_validate).grid(row=row, column=1, pady=12, sticky="w")

        frame.columnconfigure(1, weight=1)

    def _build_validate_tab(self, parent):
        frame = ttk.Frame(parent, padding=12)
        frame.pack(fill="both", expand=True)

        self.model_var = tk.StringVar()
        self.video_var = tk.StringVar()
        self.val_output_var = tk.StringVar(value=str(Path("runs_custom/validation").resolve()))
        self.conf_var = tk.StringVar(value="0.25")
        self.display_thresh_var = tk.StringVar(value="0.40")
        self.save_thresh_var = tk.StringVar(value="0.50")
        self.frame_skip_var = tk.StringVar(value="0")
        self.line_width_var = tk.StringVar(value="2")
        self.font_scale_var = tk.StringVar(value="0.7")
        self.show_window_var = tk.BooleanVar(value=True)
        self.save_video_var = tk.BooleanVar(value=True)
        self.use_tracking_var = tk.BooleanVar(value=True)
        self.tracker_yaml_var = tk.StringVar(value="bytetrack.yaml")

        row = 0
        self._path_row(frame, row, "Model (.pt)", self.model_var, self._pick_model); row += 1
        self._path_row(frame, row, "Video", self.video_var, self._pick_video); row += 1
        self._path_row(frame, row, "Validation output directory", self.val_output_var, self._pick_val_output_dir, directory=True); row += 1
        self._entry_row(frame, row, "Base inference threshold", self.conf_var); row += 1
        self._entry_row(frame, row, "Display threshold", self.display_thresh_var); row += 1
        self._entry_row(frame, row, "Save threshold", self.save_thresh_var); row += 1
        self._entry_row(frame, row, "Frame skip", self.frame_skip_var); row += 1
        self._entry_row(frame, row, "Box line width", self.line_width_var); row += 1
        self._entry_row(frame, row, "Font scale", self.font_scale_var); row += 1
        self._entry_row(frame, row, "Tracker config", self.tracker_yaml_var); row += 1

        ttk.Checkbutton(frame, text="Show live validation window", variable=self.show_window_var).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Checkbutton(frame, text="Save annotated video", variable=self.save_video_var).grid(row=row, column=1, sticky="w", pady=6)
        row += 1
        ttk.Checkbutton(frame, text="Use object tracking", variable=self.use_tracking_var).grid(row=row, column=0, sticky="w", pady=6)
        row += 1

        ttk.Button(frame, text="Start Validation", command=self.start_validation).grid(row=row, column=0, sticky="w", pady=12)

        frame.columnconfigure(1, weight=1)

    def _build_log_panel(self):
        wrap = ttk.LabelFrame(self, text="Status / Logs", padding=8)
        wrap.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.log_text = scrolledtext.ScrolledText(wrap, height=16, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    def _path_row(self, parent, row, label, variable, command, directory=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=6)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, pady=6)

    def _entry_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=6)

    def _format_model_option(self, option):
        model_name, description = option
        return f"{model_name} ({description})"

    def _parse_model_option(self, value: str) -> str:
        for model_name, _ in YOLO_MODEL_OPTIONS:
            if value.startswith(model_name):
                return model_name
        return value.strip()

    def _base_model_row(self, parent, row):
        ttk.Label(parent, text="Base model").grid(row=row, column=0, sticky="w", pady=6)
        model_values = [self._format_model_option(option) for option in YOLO_MODEL_OPTIONS]
        combo = ttk.Combobox(
            parent,
            textvariable=self.base_model_display_var,
            state="readonly",
            values=model_values,
        )
        combo.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=6)
        combo.bind("<<ComboboxSelected>>", self._on_model_selected)

    def _on_model_selected(self, event=None):
        self.base_model_var.set(self._parse_model_option(self.base_model_display_var.get()))

    def log(self, message: str):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.update_idletasks()

    def _pick_dataset(self):
        path = filedialog.askopenfilename(title="Select ZIP dataset", filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")])
        if path:
            self.dataset_var.set(path)
            return
        path = filedialog.askdirectory(title="Select dataset folder")
        if path:
            self.dataset_var.set(path)

    def _pick_output_dir(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_var.set(path)

    def _pick_model(self):
        path = filedialog.askopenfilename(title="Select trained model", filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Select validation video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("All files", "*.*")],
        )
        if path:
            self.video_var.set(path)

    def _pick_val_output_dir(self):
        path = filedialog.askdirectory(title="Select validation output directory")
        if path:
            self.val_output_var.set(path)

    def copy_model_to_validate(self):
        if self.best_model_path_var.get().strip():
            self.model_var.set(self.best_model_path_var.get().strip())
            self.log("Copied best model path into Validate tab.")

    def start_training(self):
        thread = threading.Thread(target=self._train_worker, daemon=True)
        thread.start()

    def start_validation(self):
        thread = threading.Thread(target=self._validate_worker, daemon=True)
        thread.start()

    def _train_worker(self):
        try:
            cfg = TrainConfig(
                dataset_path=self.dataset_var.get().strip(),
                output_dir=self.output_var.get().strip() or "runs_custom",
                base_model=self._parse_model_option(self.base_model_display_var.get()) or "yolo11n.pt",
                epochs=int(self.epochs_var.get()),
                imgsz=int(self.imgsz_var.get()),
                batch=int(self.batch_var.get()),
                class_names_csv=self.class_names_var.get().strip(),
                val_ratio=float(self.val_ratio_var.get()),
                seed=int(self.seed_var.get()),
            )
            best_model = self.pipeline.train(cfg)
            self.best_model_path_var.set(str(best_model))
            self.model_var.set(str(best_model))
            export_choice = self.export_format_var.get().strip().lower()
            exported = self.pipeline.export_model(str(best_model), export_choice, str(Path(cfg.output_dir) / "exports"))
            if exported:
                self.log(f"Exported model: {exported}")
            self.log("Training workflow finished successfully.")
            messagebox.showinfo("Done", "Training completed successfully.")
        except (PipelineError, ValueError) as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror("Unexpected error", str(e))

    def _validate_worker(self):
        try:
            cfg = ValidateConfig(
                model_path=self.model_var.get().strip(),
                video_path=self.video_var.get().strip(),
                output_dir=self.val_output_var.get().strip() or "runs_custom/validation",
                conf_threshold=float(self.conf_var.get()),
                display_threshold=float(self.display_thresh_var.get()),
                save_threshold=float(self.save_thresh_var.get()),
                frame_skip=int(self.frame_skip_var.get()),
                line_width=int(self.line_width_var.get()),
                font_scale=float(self.font_scale_var.get()),
                show_window=bool(self.show_window_var.get()),
                save_video=bool(self.save_video_var.get()),
                use_tracking=bool(self.use_tracking_var.get()),
                tracker_yaml=self.tracker_yaml_var.get().strip() or "bytetrack.yaml",
            )
            outputs = self.pipeline.validate_video(cfg)
            for key, path in outputs.items():
                if path:
                    self.log(f"{key.upper()}: {path}")
            self.log("Validation workflow finished successfully.")
            messagebox.showinfo("Done", "Validation completed successfully.")
        except (PipelineError, ValueError) as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.log(traceback.format_exc())
            messagebox.showerror("Unexpected error", str(e))


if __name__ == "__main__":
    App().mainloop()
