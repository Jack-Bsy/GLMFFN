import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import GLMFFN


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs") / "glmffn"
CURRENT_DIR = Path(__file__).resolve().parent
PRETRAINED_DIR = CURRENT_DIR / "pretrained_weights"

RGB_COLOR_TO_LABEL = {
    (0, 0, 0): 0,
    (0, 0, 150): 1,
    (0, 85, 0): 2,
}
IGNORE_LABEL = 255
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLMFFN for semantic segmentation.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--focal_ratio", type=float, default=0.5)
    parser.add_argument("--dice_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--save_vis_every", type=int, default=10)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--resnet_pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vmamba_pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vmamba_model_name", default="vmamba_tiny_s1l8")
    return parser.parse_args()


def prepare_pretrained_cache():
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(PRETRAINED_DIR)
    os.environ["HF_HOME"] = str(PRETRAINED_DIR / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(PRETRAINED_DIR / "hf" / "hub")
    os.environ["TIMM_HOME"] = str(PRETRAINED_DIR / "timm")
    torch.hub.set_dir(str(PRETRAINED_DIR / "hub"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_mask_as_label(mask_path, ignore_label=IGNORE_LABEL):
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    if raw.ndim == 2:
        label = raw.astype(np.uint8)
        unique = set(np.unique(label).tolist())
        if unique.issubset({0, 255}):
            label = (label > 0).astype(np.uint8)
        else:
            label[label == 255] = ignore_label
        return label

    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    label = np.full(mask_rgb.shape[:2], ignore_label, dtype=np.uint8)
    known = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for color, class_id in RGB_COLOR_TO_LABEL.items():
        pixels = np.all(mask_rgb == color, axis=2)
        label[pixels] = class_id
        known |= pixels
    white = np.all(mask_rgb == (255, 255, 255), axis=2)
    known |= white
    if (~known).any():
        print(f"warning: {mask_path.name} has {int((~known).sum())} unknown-color pixels; set to ignore.")
    return label


def infer_num_classes(data_root, ignore_label=IGNORE_LABEL):
    max_label = 0
    for split in ("train", "val", "test"):
        mask_dir = data_root / split / "mask"
        if not mask_dir.exists():
            continue
        for mask_path in mask_dir.glob("*.png"):
            label = read_mask_as_label(mask_path, ignore_label=ignore_label)
            valid = label != ignore_label
            if valid.any():
                max_label = max(max_label, int(label[valid].max()))
    return max_label + 1


class SegDataset(Dataset):
    def __init__(self, data_root, split, augment=False, ignore_label=IGNORE_LABEL):
        self.image_dir = Path(data_root) / split / "img"
        self.mask_dir = Path(data_root) / split / "mask"
        self.augment = augment
        self.ignore_label = ignore_label
        self.names = sorted(
            p.name for p in self.image_dir.glob("*.png") if (self.mask_dir / p.name).exists()
        )
        if not self.names:
            raise FileNotFoundError(f"No paired PNG files found under {self.image_dir} and {self.mask_dir}")

    def __len__(self):
        return len(self.names)

    def light_augment(self, image, label):
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[::-1, :])
            label = np.ascontiguousarray(label[::-1, :])
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            image = np.ascontiguousarray(np.rot90(image, k))
            label = np.ascontiguousarray(np.rot90(label, k))

        image_f = image.astype(np.float32)
        contrast = 1.0 + random.uniform(-0.10, 0.10)
        brightness = random.uniform(-25.5, 25.5)
        image_f = (image_f - 127.5) * contrast + 127.5 + brightness

        gamma = 1.0 + random.uniform(-0.10, 0.10)
        image_f = 255.0 * np.power(np.clip(image_f, 0, 255) / 255.0, gamma)

        if random.random() < 0.30:
            hsv = cv2.cvtColor(np.clip(image_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-3, 3)) % 180
            hsv[:, :, 1] *= 1.0 + random.uniform(-0.10, 0.10)
            hsv[:, :, 2] *= 1.0 + random.uniform(-0.10, 0.10)
            hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
            image_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        if random.random() < 0.20:
            image_f += np.random.normal(0, 5.0, image_f.shape).astype(np.float32)
        image_f = np.clip(image_f, 0, 255).astype(np.uint8)

        if random.random() < 0.05:
            image_f = cv2.GaussianBlur(image_f, (3, 3), 0)

        return image_f, label

    def __getitem__(self, index):
        name = self.names[index]
        image_path = self.image_dir / name
        mask_path = self.mask_dir / name

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = read_mask_as_label(mask_path, ignore_label=self.ignore_label)
        if label.shape[:2] != image.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, label = self.light_augment(image, label)

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label.astype(np.int64)).long()
        return image, label, name


def build_model(args, num_classes):
    return GLMFFN(
        num_classes=num_classes,
        in_channels=3,
        resnet_pretrained=args.resnet_pretrained,
        vmamba_pretrained=args.vmamba_pretrained,
        vmamba_model_name=args.vmamba_model_name,
    )


def dynamic_class_weights(target, num_classes, ignore_label=IGNORE_LABEL, eps=1e-6):
    valid = target != ignore_label
    if not valid.any():
        return torch.ones(num_classes, device=target.device)
    counts = torch.bincount(target[valid].reshape(-1), minlength=num_classes).float().to(target.device)
    present = counts > 0
    weights = torch.zeros(num_classes, device=target.device)
    if present.any():
        present_count = present.sum().float()
        weights[present] = counts[present].sum() / (present_count * counts[present].clamp_min(eps))
        weights[present] = weights[present] / weights[present].mean().clamp_min(eps)
    return weights


def focal_loss(logits, target, class_weights, gamma=2.0, ignore_label=IGNORE_LABEL, eps=1e-6):
    valid = target != ignore_label
    if not valid.any():
        return logits.sum() * 0.0
    safe_target = target.clone()
    safe_target[~valid] = 0
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    target_1c = safe_target.unsqueeze(1)
    log_pt = log_probs.gather(1, target_1c).squeeze(1)
    pt = probs.gather(1, target_1c).squeeze(1).clamp(min=eps, max=1.0 - eps)
    pixel_weights = class_weights[safe_target]
    return (-pixel_weights * ((1.0 - pt) ** gamma) * log_pt)[valid].mean()


def dice_loss(logits, target, class_weights, ignore_label=IGNORE_LABEL, eps=1e-6):
    num_classes = logits.shape[1]
    valid = target != ignore_label
    if not valid.any():
        return logits.sum() * 0.0
    safe_target = target.clone()
    safe_target[~valid] = 0
    probs = F.softmax(logits, dim=1) * valid.unsqueeze(1).float()
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1).float()

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    present = one_hot.sum(dims) > 0
    if not present.any():
        return 1.0 - dice.mean()

    weights = class_weights.clone()
    weights[~present] = 0
    if weights.sum() <= eps:
        weights = present.float()
    weights = weights / weights.sum().clamp_min(eps)
    return (weights * (1.0 - dice)).sum()


def combined_loss(logits, target, args):
    class_weights = dynamic_class_weights(target, logits.shape[1], ignore_label=args.ignore_label)
    fl = focal_loss(logits, target, class_weights, gamma=args.gamma, ignore_label=args.ignore_label)
    dl = dice_loss(logits, target, class_weights, ignore_label=args.ignore_label)
    return args.focal_ratio * fl + args.dice_ratio * dl, fl.detach(), dl.detach(), class_weights.detach()


@torch.no_grad()
def update_confusion_matrix(confusion, logits, target, num_classes, ignore_label=IGNORE_LABEL):
    pred = logits.argmax(dim=1)
    valid = target != ignore_label
    if not valid.any():
        return confusion
    target_valid = target[valid].reshape(-1)
    pred_valid = pred[valid].reshape(-1)
    keep = (target_valid >= 0) & (target_valid < num_classes) & (pred_valid >= 0) & (pred_valid < num_classes)
    if not keep.any():
        return confusion
    indices = target_valid[keep] * num_classes + pred_valid[keep]
    hist = torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion + hist.detach().cpu()


def _divide(numerator, denominator, active):
    numerator = numerator.astype(np.float64)
    denominator = denominator.astype(np.float64)
    out = np.zeros_like(numerator, dtype=np.float64)
    valid = active & (denominator > 0)
    out[valid] = numerator[valid] / denominator[valid]
    out[~active] = np.nan
    return out


def metrics_from_confusion(confusion):
    cm = confusion.numpy().astype(np.float64)
    num_classes = cm.shape[0]
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    total = cm.sum()
    tn = total - tp - fp - fn
    active = (tp + fp + fn) > 0

    iou = _divide(tp, tp + fp + fn, active)
    recall = _divide(tp, tp + fn, active)
    precision = _divide(tp, tp + fp, active)
    f1 = _divide(2 * precision * recall, precision + recall, active & ~np.isnan(precision) & ~np.isnan(recall))
    acc = _divide(tp + tn, np.full(num_classes, total), active)

    def nanmean(values):
        return float(np.nanmean(values)) if not np.all(np.isnan(values)) else 0.0

    metrics = {
        "miou": nanmean(iou),
        "mrecall": nanmean(recall),
        "mf1": nanmean(f1),
        "mprecision": nanmean(precision),
        "macc": nanmean(acc),
    }
    for class_id in range(num_classes):
        metrics[f"class{class_id}_iou"] = 0.0 if np.isnan(iou[class_id]) else float(iou[class_id])
        metrics[f"class{class_id}_recall"] = 0.0 if np.isnan(recall[class_id]) else float(recall[class_id])
        metrics[f"class{class_id}_f1"] = 0.0 if np.isnan(f1[class_id]) else float(f1[class_id])
        metrics[f"class{class_id}_precision"] = 0.0 if np.isnan(precision[class_id]) else float(precision[class_id])
        metrics[f"class{class_id}_acc"] = 0.0 if np.isnan(acc[class_id]) else float(acc[class_id])
    return metrics


def label_to_color(label, ignore_label=IGNORE_LABEL):
    safe_label = label.copy()
    ignore = safe_label == ignore_label
    safe_label[ignore] = 0
    colors = np.array(
        [
            [0, 0, 0],
            [0, 0, 150],
            [0, 85, 0],
        ],
        dtype=np.uint8,
    )
    color = colors[safe_label % len(colors)]
    color[ignore] = np.array([255, 255, 255], dtype=np.uint8)
    return color


def add_metrics_to_row(row, prefix, metrics, num_classes):
    for key in ("miou", "mrecall", "mf1", "mprecision", "macc"):
        row[f"{prefix}_{key}"] = metrics[key]
    for class_id in range(num_classes):
        for key in ("iou", "recall", "f1", "precision", "acc"):
            row[f"{prefix}_class{class_id}_{key}"] = metrics[f"class{class_id}_{key}"]


def write_log(log_path, rows):
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        preferred = ["epoch", "lr", "train_loss", "train_focal", "train_dice", "val_loss", "val_focal", "val_dice"]
        metric_keys = sorted({key for row in rows for key in row if key not in preferred})
        writer = csv.DictWriter(f, fieldnames=preferred + metric_keys)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def save_visuals(model, loader, device, output_dir, epoch, ignore_label, max_images=8):
    model.eval()
    vis_dir = output_dir / "vis" / f"epoch_{epoch:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    mean = IMAGENET_MEAN.reshape(1, 1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 1, 3)

    for images, labels, names in loader:
        images_device = images.to(device)
        logits = model(images_device)
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        images_np = images.numpy().transpose(0, 2, 3, 1)
        images_np = np.clip((images_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
        labels_np = labels.numpy().astype(np.uint8)

        for image, label, pred, name in zip(images_np, labels_np, preds, names):
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(label_to_color(label, ignore_label), cv2.COLOR_RGB2BGR)
            pred_bgr = cv2.cvtColor(label_to_color(pred, ignore_label), cv2.COLOR_RGB2BGR)
            overlay = image_bgr.copy()
            fg = pred > 0
            blended = cv2.addWeighted(image_bgr, 0.55, pred_bgr, 0.45, 0)
            overlay[fg] = blended[fg]
            stem = Path(name).stem
            cv2.imwrite(str(vis_dir / f"{stem}_image.png"), image_bgr)
            cv2.imwrite(str(vis_dir / f"{stem}_target.png"), target_bgr)
            cv2.imwrite(str(vis_dir / f"{stem}_pred.png"), pred_bgr)
            cv2.imwrite(str(vis_dir / f"{stem}_overlay.png"), overlay)
            saved += 1
            if saved >= max_images:
                return


def run_epoch(model, loader, optimizer, scaler, device, args, num_classes, train=True):
    model.train(train)
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    batches = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    desc = "train" if train else "eval"

    for images, labels, _ in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=args.amp):
                logits = model(images)
                loss, fl, dl, class_weights = combined_loss(logits, labels, args)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        confusion = update_confusion_matrix(confusion, logits.detach(), labels, num_classes, ignore_label=args.ignore_label)
        total_loss += loss.item()
        total_focal += fl.item()
        total_dice += dl.item()
        batches += 1

    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "loss": total_loss / max(1, batches),
            "focal": total_focal / max(1, batches),
            "dice": total_dice / max(1, batches),
        }
    )
    return metrics


def save_checkpoint(path, model, optimizer, epoch, num_classes, best_miou, args):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_name": "glmffn",
            "optimizer": optimizer.state_dict(),
            "num_classes": num_classes,
            "best_miou": best_miou,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    prepare_pretrained_cache()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = args.num_classes or infer_num_classes(args.data_root, ignore_label=args.ignore_label)
    print(f"data_root: {args.data_root}")
    print(f"output_dir: {args.output_dir}")
    print(f"num_classes: {num_classes}")
    print(f"resnet_pretrained: {args.resnet_pretrained}")
    print(f"vmamba_model_name: {args.vmamba_model_name}")
    print(f"vmamba_pretrained: {args.vmamba_pretrained}")

    train_set = SegDataset(args.data_root, "train", augment=not args.no_aug, ignore_label=args.ignore_label)
    val_set = SegDataset(args.data_root, "val", augment=False, ignore_label=args.ignore_label)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = build_model(args, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler(enabled=args.amp)

    best_miou = -1.0
    no_improve_epochs = 0
    logs = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args, num_classes, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, args, num_classes, train=False)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_focal": train_metrics["focal"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_focal": val_metrics["focal"],
            "val_dice": val_metrics["dice"],
        }
        add_metrics_to_row(row, "train", train_metrics, num_classes)
        add_metrics_to_row(row, "val", val_metrics, num_classes)
        logs.append(row)
        write_log(args.output_dir / "train_log.csv", logs)
        save_checkpoint(args.output_dir / "last_glmffn.pth", model, optimizer, epoch, num_classes, best_miou, args)

        print(
            f"lr={current_lr:.6g} | "
            f"train loss={train_metrics['loss']:.4f}, mIoU={train_metrics['miou']:.4f}, mF1={train_metrics['mf1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, mIoU={val_metrics['miou']:.4f}, mF1={val_metrics['mf1']:.4f}"
        )

        if val_metrics["miou"] > best_miou + args.min_delta:
            best_miou = val_metrics["miou"]
            no_improve_epochs = 0
            save_checkpoint(args.output_dir / "best_glmffn.pth", model, optimizer, epoch, num_classes, best_miou, args)
            print(f"saved best_glmffn.pth, val_mIoU={best_miou:.4f}")
        else:
            no_improve_epochs += 1
            print(f"early stopping counter: {no_improve_epochs}/{args.early_stopping_patience}")

        if args.save_vis_every > 0 and (epoch == 1 or epoch % args.save_vis_every == 0 or epoch == args.epochs):
            save_visuals(model, val_loader, device, args.output_dir, epoch, args.ignore_label)

        if no_improve_epochs >= args.early_stopping_patience:
            print(f"early stopping at epoch {epoch}; best val_mIoU={best_miou:.4f}")
            break

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
