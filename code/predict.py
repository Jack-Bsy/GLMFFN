import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import GLMFFN
from train import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    IGNORE_LABEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    add_metrics_to_row,
    label_to_color,
    metrics_from_confusion,
    prepare_pretrained_cache,
    read_mask_as_label,
    update_confusion_matrix,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict masks with a trained GLMFFN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "best_glmffn.pth")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image_dir", type=Path, default=None)
    parser.add_argument("--mask_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "test_predictions")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pad_divisor", type=int, default=32)
    parser.add_argument("--resnet_pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vmamba_pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vmamba_model_name", default="vmamba_tiny_s1l8")
    return parser.parse_args()


def list_images(image_dir):
    return sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.name.lower(),
    )


def find_mask(image_path, mask_dir):
    if mask_dir is None or not mask_dir.exists():
        return None
    candidates = [mask_dir / image_path.name]
    candidates.extend(mask_dir / f"{image_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def preprocess_image(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    return image_bgr, tensor


def pad_tensor(tensor, divisor):
    if divisor <= 1:
        return tensor, tensor.shape[-2], tensor.shape[-1]
    height, width = tensor.shape[-2:]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, height, width


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return checkpoint, state_dict


def checkpoint_value(checkpoint, key, fallback):
    if isinstance(checkpoint, dict):
        value = checkpoint.get(key)
        if value is not None:
            return value
        args = checkpoint.get("args")
        if isinstance(args, dict) and args.get(key) is not None:
            return args[key]
    return fallback


def build_loaded_model(args, device):
    checkpoint, state_dict = load_checkpoint(args.checkpoint)
    num_classes = int(checkpoint_value(checkpoint, "num_classes", args.num_classes))
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model = GLMFFN(
        num_classes=num_classes,
        in_channels=3,
        resnet_pretrained=args.resnet_pretrained,
        vmamba_pretrained=args.vmamba_pretrained,
        vmamba_model_name=saved_args.get("vmamba_model_name", args.vmamba_model_name),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"warning: load_state_dict missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  first missing keys: {missing[:5]}")
        if unexpected:
            print(f"  first unexpected keys: {unexpected[:5]}")
    model.to(device)
    model.eval()
    return model, num_classes


def save_prediction(output_dir, image_path, image_bgr, label, pred, ignore_label):
    pred_dir = output_dir / "pred_masks"
    vis_dir = output_dir / "visuals"
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    pred_bgr = cv2.cvtColor(label_to_color(pred, ignore_label), cv2.COLOR_RGB2BGR)
    overlay = image_bgr.copy()
    foreground = pred > 0
    blended = cv2.addWeighted(image_bgr, 0.55, pred_bgr, 0.45, 0)
    overlay[foreground] = blended[foreground]

    stem = image_path.stem
    cv2.imwrite(str(pred_dir / f"{stem}_pred.png"), pred_bgr)
    cv2.imwrite(str(vis_dir / f"{stem}_image.png"), image_bgr)
    cv2.imwrite(str(vis_dir / f"{stem}_pred.png"), pred_bgr)
    cv2.imwrite(str(vis_dir / f"{stem}_overlay.png"), overlay)

    if label is not None:
        label_bgr = cv2.cvtColor(label_to_color(label, ignore_label), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_dir / f"{stem}_target.png"), label_bgr)

        valid = label != ignore_label
        error = np.zeros_like(image_bgr)
        error[np.logical_and(pred > 0, np.logical_and(label == 0, valid))] = (0, 0, 255)
        error[np.logical_and(pred == 0, np.logical_and(label > 0, valid))] = (255, 0, 0)
        error[np.logical_and(pred != label, np.logical_and(pred > 0, np.logical_and(label > 0, valid)))] = (0, 255, 255)
        error[~valid] = (255, 255, 255)
        cv2.imwrite(str(vis_dir / f"{stem}_error.png"), error)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return None
    fieldnames = list(rows[0].keys())
    extra = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extra)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


@torch.no_grad()
def predict(args):
    prepare_pretrained_cache()
    device = torch.device(args.device)
    image_dir = args.image_dir or args.data_root / args.split / "img"
    mask_dir = args.mask_dir or args.data_root / args.split / "mask"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir does not exist: {image_dir}")

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")

    model, num_classes = build_loaded_model(args, device)
    print(f"checkpoint: {args.checkpoint}")
    print(f"image_dir: {image_dir}")
    print(f"mask_dir: {mask_dir if mask_dir.exists() else 'not found; metrics disabled'}")
    print(f"output_dir: {args.output_dir}")
    print(f"num_classes: {num_classes}")

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    per_image_rows = []
    has_metrics = False

    for image_path in tqdm(images, desc="predict"):
        image_bgr, tensor = preprocess_image(image_path)
        tensor, old_h, old_w = pad_tensor(tensor.to(device), args.pad_divisor)
        logits = model(tensor)[:, :, :old_h, :old_w]
        if logits.shape[-2:] != image_bgr.shape[:2]:
            logits = F.interpolate(logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        mask_path = find_mask(image_path, mask_dir)
        label = None
        row = {"image": image_path.name, "prediction": f"pred_masks/{image_path.stem}_pred.png"}
        if mask_path is not None:
            label = read_mask_as_label(mask_path, ignore_label=args.ignore_label)
            if label.shape[:2] != image_bgr.shape[:2]:
                label = cv2.resize(label, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            label_tensor = torch.from_numpy(label.astype(np.int64)).unsqueeze(0).long()
            sample_confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
            sample_confusion = update_confusion_matrix(
                sample_confusion,
                logits.detach().cpu(),
                label_tensor,
                num_classes,
                ignore_label=args.ignore_label,
            )
            confusion += sample_confusion
            has_metrics = True
            row["mask"] = mask_path.name
            add_metrics_to_row(row, "image", metrics_from_confusion(sample_confusion), num_classes)

        per_image_rows.append(row)
        save_prediction(args.output_dir, image_path, image_bgr, label, pred, args.ignore_label)

    write_csv(args.output_dir / "per_image_metrics.csv", per_image_rows)
    if has_metrics:
        summary = {
            "checkpoint": str(args.checkpoint),
            "num_images": len(images),
            "num_classes": num_classes,
        }
        metrics = metrics_from_confusion(confusion)
        add_metrics_to_row(summary, "all", metrics, num_classes)
        write_csv(args.output_dir / "metrics.csv", [summary])
        print(f"mIoU={metrics['miou']:.4f}, mF1={metrics['mf1']:.4f}")
        print(f"metrics: {args.output_dir / 'metrics.csv'}")
    print(f"predictions: {args.output_dir}")


def main():
    predict(parse_args())


if __name__ == "__main__":
    main()
