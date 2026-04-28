# Plant Disease Detection & Crop Yield Prediction

An integrated deep learning framework for plant disease detection and crop yield prediction using Faster RCNN, 38 CNN models, and XGBoost.

## Setup

```bash
conda create -n tf-gpu python=3.9 -y
conda activate tf-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
pip install tensorflow==2.10
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter notebook opencv-python pillow ipywidgets torch torchvision groundingdino-py
```

## Dataset

Download from Kaggle and place in `data/`:
- Disease images: https://www.kaggle.com/datasets/nafishamoin/bangladeshi-crops-disease-dataset
- Yield data: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset → place in `data/yield_data/`

## Run

**Train Keras models:**
```bash
conda activate tf-gpu
python src/train_keras_models.py --person zahra  # or sireen / tala
```

**Evaluate all 38 models:**
```bash
python src/evaluate_all_38.py
```

**Auto-annotate with Grounding DINO:**
```bash
# Download weights to weights/ folder first:
# https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
python src/auto_annotate.py
```

**Convert annotations to COCO format:**
```bash
python src/convert_to_coco.py
```

**Train Faster RCNN:**
```bash
python src/train_faster_rcnn_torchvision.py
```

**Evaluate Faster RCNN:**
```bash
python src/evaluate_faster_rcnn.py
```

**Yield prediction + interface:**
```bash
jupyter notebook notebooks/04_yield_prediction.ipynb
```

## Results

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| ConvNeXtXLarge | 84.2% | 0.865 |
| EfficientNetV2S | 83.3% | 0.852 |
| EfficientNetB4 | 82.9% | 0.844 |

| Detector | Annotations | mAP@0.5 |
|----------|-------------|---------|
| Manual | 280 | 35% |
| Grounding DINO | 2,256 | 84.2% |

## Git Workflow

```bash
git pull origin main
# do your work
git add .
git commit -m "your message"
git push origin main
```