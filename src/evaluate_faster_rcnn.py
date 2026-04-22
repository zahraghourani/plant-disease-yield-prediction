import torch
from torch.utils.data import DataLoader
import torchvision
from train_faster_rcnn_torchvision import CocoDetection, get_transform
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(model, data_loader, device, coco_gt):
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                img_id = targets[i]['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    xmin, ymin, xmax, ymax = box
                    w = xmax - xmin
                    h = ymax - ymin
                    results.append({
                        'image_id': img_id,
                        'category_id': label,
                        'bbox': [xmin, ymin, w, h],
                        'score': score
                    })
    if not results:
        print("No predictions found.")
        return None
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dir = 'detection_data/images'
    ann_file = 'detection_data/annotations/train_coco.json'
    dataset = CocoDetection(data_dir, ann_file, transforms=get_transform())
    coco_gt = dataset.coco  # directly from the dataset
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Load trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = len(coco_gt.getCatIds()) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('faster_rcnn_model.pth', map_location=device))
    model.to(device)

    stats = evaluate(model, loader, device, coco_gt)
    if stats is not None:
        print(f"\n🎯 mAP@0.5:0.95 = {stats[0]:.4f}")
        print(f"🎯 mAP@0.5      = {stats[1]:.4f}")