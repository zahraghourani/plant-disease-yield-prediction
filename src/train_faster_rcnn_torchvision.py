"""
FAST Faster RCNN + severity head training.
Fixes:
  - num_workers=0  (Windows multiprocessing pickle error fix)
  - collate_fn moved to module level (Windows spawn fix)
  - Explicit CUDA check with helpful message
  - Backbone frozen for speed

Run from project root:
    python src/train_faster_rcnn_fast.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from pathlib import Path


# ── MUST be at module level for Windows multiprocessing spawn ─────────────────
def collate_fn(batch):
    return tuple(zip(*batch))


# ── Dataset ───────────────────────────────────────────────────────────────────
class CocoDetection(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root       = root
        self.coco       = COCO(annotation)
        self.ids        = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id   = self.ids[index]
        ann_ids  = self.coco.getAnnIds(imgIds=img_id)
        anns     = self.coco.loadAnns(ann_ids)
        path     = self.coco.loadImgs(img_id)[0]['file_name']
        matches  = list(Path(self.root).rglob(os.path.basename(path)))
        img_path = matches[0] if matches else os.path.join(self.root, path)
        img      = Image.open(img_path).convert('RGB')

        boxes, labels = [], []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        img_area = img.size[0] * img.size[1]
        area     = ((boxes[:, 3] - boxes[:, 1]) *
                    (boxes[:, 2] - boxes[:, 0])) if len(boxes) else torch.zeros(0)
        iscrowd  = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([img_id]),
            'area':     area,
            'iscrowd':  iscrowd,
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform():
    return T.Compose([T.ToTensor()])


# ── Severity Head ─────────────────────────────────────────────────────────────
class SeverityHead(nn.Module):
    def __init__(self, in_features=1024):
        super().__init__()
        self.fc1     = nn.Linear(in_features, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(256, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.dropout(F.relu(self.fc1(x)))))


# ── Severity-Aware Faster RCNN ────────────────────────────────────────────────
class SeverityAwareFasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.faster_rcnn = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_f = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(
            in_f, num_classes
        )
        # Freeze backbone — only train RPN + heads
        for param in self.faster_rcnn.backbone.parameters():
            param.requires_grad = False

        in_sev = self.faster_rcnn.roi_heads.box_head.fc7.out_features
        self.severity_head = SeverityHead(in_sev)

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            losses = self.faster_rcnn(images, targets)
            losses['severity_loss'] = self._severity_loss(images, targets)
            return losses
        else:
            detections = self.faster_rcnn(images)
            return self._add_severity(detections, images)

    def _severity_loss(self, images, targets):
        all_roi, gt_sevs = [], []
        for img, target in zip(images, targets):
            boxes = target['boxes']
            if len(boxes) == 0:
                continue
            img_area = img.shape[-2] * img.shape[-1]
            areas    = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
            gt_sev   = float((areas.sum() / img_area).clamp(0, 1))
            feat = self.faster_rcnn.backbone(img.unsqueeze(0))
            roi  = self.faster_rcnn.roi_heads.box_roi_pool(
                feat, [boxes], [img.shape[-2:]]
            )
            roi  = self.faster_rcnn.roi_heads.box_head(roi)
            all_roi.append(roi)
            gt_sevs.extend([gt_sev] * len(boxes))

        if not all_roi:
            return torch.tensor(0.0, device=images[0].device)
        roi_cat = torch.cat(all_roi, dim=0)
        pred    = self.severity_head(roi_cat)
        gt      = torch.tensor(gt_sevs, device=pred.device).unsqueeze(1)
        return F.mse_loss(pred, gt)

    def _add_severity(self, detections, images):
        for det, img in zip(detections, images):
            boxes = det['boxes']
            if len(boxes) == 0:
                det['severities'] = torch.zeros(0, device=img.device)
                continue
            feat = self.faster_rcnn.backbone(img.unsqueeze(0))
            roi  = self.faster_rcnn.roi_heads.box_roi_pool(
                feat, [boxes], [img.shape[-2:]]
            )
            roi  = self.faster_rcnn.roi_heads.box_head(roi)
            det['severities'] = self.severity_head(roi).squeeze(1)
        return detections


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── GPU check ─────────────────────────────────────────────────────────────
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("\n*** GPU not detected — running on CPU ***")
        print("To force GPU, run this first in PowerShell:")
        print("  $env:CUDA_VISIBLE_DEVICES='0'")
        print("Then re-run this script.\n")
        device = torch.device('cpu')

    data_dir = 'data/Crop___DIsease'
    ann_file = 'detection_data/annotations/train_coco.json'

    dataset = CocoDetection(data_dir, ann_file, transforms=get_transform())
    indices = torch.randperm(len(dataset)).tolist()
    split   = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    val_ds   = torch.utils.data.Subset(dataset, indices[split:])

    # num_workers=0 fixes Windows pickle/spawn error
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        num_workers=0,           # ← KEY FIX for Windows
        collate_fn=collate_fn    # ← module-level function, not lambda
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    num_classes = len(dataset.coco.getCatIds()) + 1
    print(f"\nDataset : {len(dataset)} images, {num_classes} classes")
    print(f"Train   : {len(train_ds)}  |  Val: {len(val_ds)}")

    model = SeverityAwareFasterRCNN(num_classes=num_classes)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params  : {trainable:,} trainable / {total:,} total "
          f"(backbone frozen)")

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.1
    )

    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (fp16) enabled — training will be faster\n")

    num_epochs = 5
    best_loss  = float('inf')
    out_dir    = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path   = out_dir / "faster_rcnn_with_severity.pth"

    for epoch in range(num_epochs):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    det_loss  = sum(v for k, v in loss_dict.items()
                                    if k != 'severity_loss')
                    sev_loss  = loss_dict.get(
                        'severity_loss', torch.tensor(0.0, device=device))
                    total_loss = det_loss + sev_loss
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict  = model(images, targets)
                det_loss   = sum(v for k, v in loss_dict.items()
                                 if k != 'severity_loss')
                sev_loss   = loss_dict.get(
                    'severity_loss', torch.tensor(0.0, device=device))
                total_loss = det_loss + sev_loss
                total_loss.backward()
                optimizer.step()

            epoch_loss  += total_loss.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(f"  [{epoch+1}/{num_epochs}] "
                      f"batch {batch_idx:3d}/{len(train_loader)}  "
                      f"loss={total_loss.item():.4f}  "
                      f"det={det_loss.item():.4f}  "
                      f"sev={sev_loss.item():.4f}")

        avg = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}  avg_loss={avg:.4f}")
        scheduler.step()

        if avg < best_loss:
            best_loss = avg
            torch.save({
                'model_state_dict': model.state_dict(),
                'severity_head':    model.severity_head.state_dict(),
                'num_classes':      num_classes,
                'epoch':            epoch + 1,
                'loss':             avg,
            }, str(out_path))
            print(f"  ✓ Saved best checkpoint → {out_path}\n")

    print(f"Done. Best loss={best_loss:.4f}")
    print(f"Checkpoint: {out_path}")


if __name__ == '__main__':
    main()