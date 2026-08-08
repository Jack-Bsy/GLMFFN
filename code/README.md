## Model
- Main encoder: ImageNet-pretrained ResNet34.
- Auxiliary encoder: VSSBranch with patch embedding, patch merging, and VSS blocks.
- Fusion: Multi-Scale Adaptive Fusion Module (MSAFM).
- Refinement: Multi-pooling Channel-spatial Feature Refinement Module (MCFRM).
- Decoder: Three-stage decoder with skip-feature concatenation and a segmentation head.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Format

Prepare the dataset as:

```text
data/
  train/
    img/
    mask/
  val/
    img/
    mask/
  test/
    img/
    mask/
```

Images and masks should use matching file names. RGB masks are mapped by default as:

```text
(0, 0, 0)   -> class 0
(0, 0, 150) -> class 1
(0, 85, 0)  -> class 2
```

## Training

```bash
python train.py --data_root data --output_dir outputs/glmffn --num_classes 3
```

Useful options:

```bash
python train.py \
  --data_root /path/to/dataset \
  --output_dir outputs/glmffn \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4
```

The best checkpoint is saved as:

```text
outputs/glmffn/best_glmffn.pth
```

## Prediction

```bash
python predict.py \
  --checkpoint outputs/glmffn/best_glmffn.pth \
  --data_root data \
  --split test \
  --output_dir outputs/glmffn/test_predictions
```

Prediction outputs include colored masks, overlays, optional error maps, and CSV metrics when ground-truth masks are available.