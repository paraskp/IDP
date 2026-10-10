"""
Train Road Landmarks YOLO Model.
Automates training of a lightweight YOLOv8-nano model to detect stationary
road infrastructure (dashed lane lines, zebra stripes, turn markings, poles).
Can be run on Google Colab (free GPU) or locally.
"""

import argparse
from pathlib import Path
import shutil


def train_model(
    data_yaml: Path,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    base_model: str = "yolov8n.pt",
    device: str = "0",  # "0" for GPU, "cpu" for CPU
    export_dir: Path = Path("models"),
):
    """
    Trains YOLOv8 on the stationary landmark dataset and exports weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed. Install via: pip install ultralytics")
        return

    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at: {data_yaml}")
        return

    print(f"Loading base model: {base_model}...")
    model = YOLO(base_model)

    print(f"Starting training on {data_yaml} for {epochs} epochs...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=15,
        device=device,
        save=True,
        project="runs/detect",
        name="road_landmarks",
    )

    # Locate and copy best.pt to target models directory
    best_weights = Path("runs/detect/road_landmarks/weights/best.pt")
    if best_weights.exists():
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        target_path = export_dir / "road_landmarks.pt"
        shutil.copy(best_weights, target_path)
        print(f"\nTraining Complete! Exported best weights to: {target_path}")
    else:
        print("Training finished, but best.pt could not be located automatically.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for stationary road landmark detection.")
    parser.add_argument("--data", "-d", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Training device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained base model (default: yolov8n.pt)")

    args = parser.parse_args()
    train_model(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        base_model=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
