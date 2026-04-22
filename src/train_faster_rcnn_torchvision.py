import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from pathlib import Path

# ---------- Custom Dataset using COCO JSON ----------
class CocoDetection(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        matches = list(Path(self.root).rglob(os.path.basename(path)))
        img_path = matches[0] if matches else os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')
        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform():
    return T.Compose([T.ToTensor()])

# ---------- Training setup ----------
def main():
    data_dir = 'data/Crop___Disease'
    ann_file = 'detection_data/annotations/train_coco.json'
    dataset = CocoDetection(data_dir, ann_file, transforms=get_transform())
    # Split train/val (80/20)
    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, indices[:split])
    val_dataset = torch.utils.data.Subset(dataset, indices[split:])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Load pre-trained Faster R-CNN with ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = len(dataset.coco.getCatIds()) + 1  # +1 for background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        # Optional validation step (skip for speed)
        model.eval()

    # Save model
    torch.save(model.state_dict(), 'faster_rcnn_model.pth')
    print("Model saved to faster_rcnn_model.pth")

if __name__ == '__main__':
    main()