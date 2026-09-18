# README — Full Three-Stage Pipeline (v3, Post-Leakage-Fix)

This is the complete guide for training and combining all three stages
of the pipeline: CNN classification, Faster RCNN detection, and XGBoost
yield prediction. Supersedes any earlier README.

---

## 0. Background — bugs found and fixed so far

1. **Train/val/test conflation (CNN stage).** The original script
   validated checkpoints on the same data used for final "test"
   metrics. Fixed by adding a real, separate held-out test set.

2. **Augmented-duplicate leakage (CNN stage, the bigger one).** The
   classification dataset contains many augmented copies of the same
   source photo saved as separate files (e.g. `RS_Rust 2743.JPG`,
   `RS_Rust 2743_flipLR.JPG`). A file-level split let siblings of the
   same photo land in different splits — verified 71.4% of test images
   had a sibling in train. Fixed with a group-aware split
   (`data_preprocessing.py`) that keeps all copies of one source photo
   together.

3. **Faster RCNN detection dataset — same risk, not yet confirmed.**
   The detection dataset (`detection_data/images/`) is drawn from the
   same source data and almost certainly contains the same kind of
   augmented duplicates (filenames like `RS_Rust 2743_flipLR.JPG`
   already appear inside `detection_data/annotations/`). **This needs
   to be checked before trusting the current Faster RCNN mAP
   numbers** — see Stage 2 below.

4. **XGBoost — not affected.** It trains on the FAO climate/yield
   dataset (country-year records), which has no relationship to leaf
   photos or the CNN/detection datasets. Its existing 80/20 split is
   fine as-is.

Document all of this explicitly in the paper as a "Changes from the
previous version" paragraph — reviewers should never see an
unexplained accuracy shift without an explanation.

---

## 1. Environment setup

Conda env: `tf-gpu` for the CNN stage (TensorFlow 2.10). Faster RCNN
uses a separate PyTorch environment (`detectron2_env`, per project
notes) — confirm which env has torchvision installed before running
detection scripts.

Always call interpreters by full path rather than relying on `conda
activate`, which has not reliably persisted in this environment:
```powershell
& "C:\Users\HPZ4-03-Adm01\miniconda3\envs\tf-gpu\python.exe" <script>
& "C:\Users\HPZ4-03-Adm01\miniconda3\envs\detectron2_env\python.exe" <script>
```

**If GPU shows `GPU: []` (CPU fallback) in a TensorFlow run:**
```powershell
$env:PATH = "C:\Users\HPZ4-03-Adm01\miniconda3\envs\tf-gpu\Library\bin;C:\Users\HPZ4-03-Adm01\miniconda3\envs\tf-gpu;$env:PATH"
```
Then verify:
```powershell
& "...\tf-gpu\python.exe" -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Do this in the same terminal session you'll train in.

---

## STAGE 1 — CNN Classification (38 models)

### 1a. Verify the dataset
```powershell
(Get-ChildItem -Path "...\data\Crop___DIsease" -Directory).Count
```
Must print **15**. Full expected class counts (31,053 total):
Corn_Common_Rust 3814, Corn_Gray_Leaf_Spot 3284, Corn_Healthy 3718,
Corn_Leaf_Blight 3816, Invalid 1563, Potato_Early_Blight 3149,
Potato_Healthy 2006, Potato_Late_Blight 3131, Rice_Brown_Spot 563,
Rice_Healthy 523, Rice_Hispa 523, Rice_Leaf_Blast 1182,
Wheat_Brown_Rust 1128, Wheat_Healthy 1497, Wheat_Yellow_Rust 1156.

### 1b. Fresh retrain setup (only if split code or dataset changed)
```powershell
Remove-Item .\splits.csv -ErrorAction SilentlyContinue
Remove-Item .\ALL_38_MODELS_COMBINED.csv -ErrorAction SilentlyContinue
Remove-Item .\test_predictions_all_38.csv -ErrorAction SilentlyContinue
mkdir results_OLD_vN, checkpoints_OLD_vN -ErrorAction SilentlyContinue
Move-Item results\*, checkpoints\* *_OLD_vN\ -ErrorAction SilentlyContinue
```
**Never delete `splits.csv` mid-way through the 38-model run** — every
model must be trained/evaluated on identical partitions.

### 1c. Sanity check before the full run
```powershell
& "...\tf-gpu\python.exe" src\train_keras_models.py --person zahra --model EfficientNetB0
& "...\tf-gpu\python.exe" src\check_duplicate_leakage.py
```
Second command must show **0% leakage** before proceeding.

### 1d. Full retrain
```powershell
& "...\tf-gpu\python.exe" src\train_keras_models.py --person zahra
& "...\tf-gpu\python.exe" src\train_keras_models.py --person sireen
& "...\tf-gpu\python.exe" src\train_keras_models.py --person tala
```
Assignments: Zahra (13) - Xception, VGG16, ResNet152, ResNet152V2,
MobileNet, MobileNetV2, EfficientNetB0-B2, EfficientNetV2B0/S/M,
ConvNeXtLarge. Sireen (12) - VGG19, ResNet50, ResNet50V2, DenseNet121/
169/201, EfficientNetB3-B5, EfficientNetV2L, ConvNeXtTiny/Small.
Tala (13) - ResNet101/101V2, InceptionV3, InceptionResNetV2,
NASNetMobile/Large, EfficientNetB6/B7, EfficientNetV2B1-B3,
ConvNeXtBase/XLarge.

### 1e. After all 38 finish
```powershell
Remove-Item .\ALL_38_MODELS_COMBINED.csv -ErrorAction SilentlyContinue
& "...\tf-gpu\python.exe" src\evaluate_all_38.py
& "...\tf-gpu\python.exe" src\generate_test_predictions.py
```
Confirm 38 final checkpoints exist, then identify the winning model
(highest accuracy) - this is relevant context for Stage 3, but note
the deployed model may differ from the most accurate one (see 3c).

---

## STAGE 2 - Faster RCNN Detection

### 2a. Annotation pipeline (already done, documented here for reference)
- **Manual**: 280 images hand-annotated in LabelImg (`detection_data/
  annotations/`), including 20 newly-added `Corn___Gray_Leaf_Spot`
  annotations
- **Automated**: Grounding DINO auto-annotated 2,256 images
  (`detection_data/auto_annotations_full/`), using `auto_annotate.py`
  with class-specific prompts, box threshold 0.35, text threshold 0.25
- **IoU validation (Item 19, done)**: `evaluate_annotation_quality.py`
  compared Grounding DINO boxes against all 280 manual images. Real
  result: mean IoU = 0.389, precision = 0.770, recall = 0.194 @
  IoU>=0.5. Root cause identified: Grounding DINO tends to return one
  box per image regardless of true lesion count (e.g. 18 manual boxes
  vs. 1 predicted on one Potato Early Blight image). Written into the
  paper with a new table + comparison figure.

### 2b. CHECK FOR THE SAME DUPLICATE LEAKAGE BEFORE TRUSTING mAP NUMBERS
Not yet done - do this before anything else in this stage. Adapt
`check_duplicate_leakage.py`'s grouping logic to scan
`detection_data/images/` and whatever train/val split file the Faster
RCNN training script uses (check `train_faster_rcnn.py` /
`train_faster_rcnn_torchvision.py` / `convert_to_coco.py` for where
the 80/20 split is actually defined - likely a COCO JSON or a file
list). If duplicates are found crossing the split, the same
group-aware fix approach applies here too, and Faster RCNN needs
retraining on a corrected split.

### 2c. Train the severity-aware Faster RCNN
```powershell
& "...\detectron2_env\python.exe" src\train_faster_rcnn.py
```
(Or `train_faster_rcnn_torchvision.py`, depending on which is current
- confirm with whoever last touched this script.) Config per the
paper: SGD lr=0.005, momentum 0.9, weight decay 5e-4, step LR
scheduler (step 2, gamma=0.1), batch size 4, 5 epochs, ResNet-50 FPN
backbone pretrained on COCO, mixed precision (fp16), frozen backbone.

### 2d. Evaluate
```powershell
& "...\detectron2_env\python.exe" src\evaluate_faster_rcnn.py
```
Reports mAP@0.5, mAP@0.5:0.95, and severity-head MAE against
ground-truth lesion coverage ratios. Original numbers (pre any
detection-leakage fix): mAP@0.5 = 61.4%, mAP@0.5:0.95 = 45.5%,
severity MAE = 0.2198 - treat these as provisional until Step 2b is
resolved.

---

## STAGE 3 - XGBoost Yield Prediction

### 3a. Train XGBoost/Random Forest on literature severity proxies
```powershell
& "...\tf-gpu\python.exe" src\yield_prediction.py
```
Uses the FAO yield dataset (Crop Yield Prediction Dataset, 15,642
country-year records), 80/20 split (12,513 train / 3,129 test,
seed=42). Severity input during training is the fixed literature
proxy per crop (Maize/Wheat: 0.25, Rice: 0.263, Potato: 0.217) - not
computed from any image, since the FAO records have no paired photos.
This part is unaffected by the CNN/detection leakage issues.

### 3b. Ablation study
```powershell
& "...\tf-gpu\python.exe" src\yield_ablation.py
```
Reproduces Table 9 (Climate only -> +crop type -> +disease severity ->
full model). Confirm the "removing severity" claim in the text matches
this table's actual numbers (0.9699 -> 0.9743, NOT "near-zero" - this
was flagged by multiple reviewers as incorrect).

### 3c. Combining Stage 1 + Stage 2 into Stage 3 - dynamic severity at inference
This is the "integration" step, used only at **inference time** (not
during XGBoost training, since training uses fixed proxies):
```
severity_dynamic = (sum of Faster RCNN lesion box areas / image area) x CNN softmax confidence
```
- The **CNN** here is whichever Stage 1 model is chosen for deployment
  (EfficientNetV2S in the original submission, chosen for its
  ROC-AUC/size/speed tradeoff - NOT necessarily the single most
  accurate model, since deployment favors efficiency). Confirm this
  choice still makes sense once Stage 1's corrected retrain finishes.
- The **Faster RCNN** is the Stage 2 detector (with detection threshold
  0.4 for boxes counted).
- This combined score is validated in `create_dynamic_severity_training_data.py`
  / `compute_severity.py` (Section 4.4 "Dynamic Severity Validation" -
  15 images per class compared against literature values). Re-run this
  once Stage 1 and Stage 2 both have corrected, verified-leak-free
  results.

**Important distinction to keep straight in the paper**: the "most
accurate" Stage 1 model (currently ConvNeXtBase/XLarge territory) and
the "deployed" Stage 1 model (EfficientNetV2S, chosen for efficiency)
can be different - the dynamic severity computation uses whichever
model is actually deployed, not necessarily the leaderboard-topping
one. State this explicitly rather than letting it look inconsistent.

---

## Known gotchas, quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `GPU: []` in training log | PATH missing tf-gpu's `Library\bin` | See Section 1 PATH fix |
| `evaluate_all_38.py` skips everything | Stale `ALL_38_MODELS_COMBINED.csv` exists | Delete it before running |
| Accuracy jump after a "fix" looks too good | Possible new leakage | Run `check_duplicate_leakage.py` (or its Stage 2 equivalent) before trusting any new number |
| Faster RCNN mAP looks unexpectedly high/low after any dataset change | Possible duplicate leakage in detection split | Check Stage 2b before trusting results |