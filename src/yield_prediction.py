"""
Plant Disease Detection + Crop Yield Prediction — Web Interface
Stage 3 TRUE integration:
  severity = learned severity head output (per-box, aggregated)
  passed directly into XGBoost as a real input feature.

Run from project root:
    python src/yield_prediction.py
"""
from __future__ import annotations

import base64
import cgi
import html
import io
import sys
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[1]
DATA_DIR        = ROOT / "data" / "yield_data" / "yield_df.csv"
IMAGE_DATA_DIR  = ROOT / "data" / "Crop___DIsease"
CHECKPOINT      = ROOT / "checkpoints" / "final_EfficientNetV2S.weights.h5"
LEADERBOARD     = ROOT / "clean_scores_all38.csv"
RCNN_SEVERITY   = ROOT / "checkpoints" / "faster_rcnn_with_severity.pth"
RCNN_PLAIN      = ROOT / "checkpoints" / "faster_rcnn_model.pth"

DISEASE_MODEL_NAME = "EfficientNetV2S"
TARGET_CROPS = ["Maize", "Wheat", "Rice, paddy", "Potatoes"]
DETECTION_THRESHOLD = 0.4
NUM_CLASSES = 15   # 14 disease classes + background

# ── Literature severity (manual mode fallback only) ───────────────────────────
DISEASE_YIELD_IMPACT = {
    "Corn___Common_Rust": 0.35, "Corn___Leaf_Blight": 0.40,
    "Corn___Healthy": 0.00, "Potato___Early_Blight": 0.20,
    "Potato___Late_Blight": 0.45, "Potato___Healthy": 0.00,
    "Rice___Brown_Spot": 0.25, "Rice___Healthy": 0.00,
    "Rice___Hispa": 0.30, "Rice___Leaf_Blast": 0.50,
    "Wheat___Brown_Rust": 0.35, "Wheat___Healthy": 0.00,
    "Wheat___Yellow_Rust": 0.40, "Invalid": 0.00, "Unknown": 0.00,
}

# Crop-average severity proxies for XGBoost training
# (FAO dataset has no images; at inference the learned severity replaces this)
CROP_AVG_SEVERITY = {
    "Maize":       (0.35 + 0.40 + 0.00) / 3,
    "Wheat":       (0.35 + 0.40 + 0.00) / 3,
    "Rice, paddy": (0.25 + 0.30 + 0.50 + 0.00) / 4,
    "Potatoes":    (0.20 + 0.45 + 0.00) / 3,
}

DISEASE_OPTIONS = {
    "Maize":       ["Corn___Common_Rust", "Corn___Leaf_Blight", "Corn___Healthy"],
    "Wheat":       ["Wheat___Brown_Rust", "Wheat___Yellow_Rust", "Wheat___Healthy"],
    "Rice, paddy": ["Rice___Brown_Spot", "Rice___Hispa",
                    "Rice___Leaf_Blast", "Rice___Healthy"],
    "Potatoes":    ["Potato___Early_Blight", "Potato___Late_Blight", "Potato___Healthy"],
}

DISEASE_TO_CROP = {
    "Corn___Common_Rust": "Maize", "Corn___Leaf_Blight": "Maize",
    "Corn___Healthy": "Maize", "Potato___Early_Blight": "Potatoes",
    "Potato___Late_Blight": "Potatoes", "Potato___Healthy": "Potatoes",
    "Rice___Brown_Spot": "Rice, paddy", "Rice___Hispa": "Rice, paddy",
    "Rice___Leaf_Blast": "Rice, paddy", "Rice___Healthy": "Rice, paddy",
    "Wheat___Brown_Rust": "Wheat", "Wheat___Yellow_Rust": "Wheat",
    "Wheat___Healthy": "Wheat", "Invalid": "Maize", "Unknown": "Maize",
}

PESTICIDE_LEVELS = {
    "None": 0.0, "Low": 1700.0, "Medium": 17500.0,
    "High": 49000.0, "Very high": 100000.0,
}


# =============================================================================
#  Dynamic severity  (USES TRAINED SEVERITY HEAD — no longer manual formula)
# =============================================================================
def compute_learned_severity(
    boxes: list, scores: list, severities: list,
    threshold: float = DETECTION_THRESHOLD,
) -> tuple[float, float, int, float]:
    """
    Uses the trained severity head output instead of manual formula.

    Aggregates per-box learned severities for boxes above detection threshold.
    Returns: (severity, lesion_ratio, n_boxes, max_severity)
    """
    total = 0.0
    n = 0
    sev_sum = 0.0
    sev_max = 0.0
    image_area = 1.0  # not needed for learned severity, but kept for compat

    for box, score, sev in zip(boxes, scores, severities):
        if score >= threshold:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            total += box_area
            n += 1
            sev_val = float(sev)
            sev_sum += sev_val
            sev_max = max(sev_max, sev_val)

    ratio = total / image_area if image_area > 0 else 0.0

    # Aggregate learned severities: mean of detected boxes, or 0 if none
    if n > 0:
        severity = sev_sum / n
    else:
        severity = 0.0

    return float(min(max(severity, 0.0), 1.0)), ratio, n, sev_max


# =============================================================================
#  Result dataclass
# =============================================================================
@dataclass
class YieldResult:
    crop: str
    disease: str
    rainfall: float
    pesticides: float
    pesticide_label: str
    temperature: float
    severity: float
    severity_method: str   # "learned" or "literature"
    base_yield: float
    adjusted_yield: float
    estimated_loss: float
    confidence: float | None = None
    image_preview: str | None = None
    n_lesion_boxes: int = 0
    lesion_ratio: float = 0.0
    max_severity: float = 0.0  # NEW: max per-box learned severity


# =============================================================================
#  Disease Localiser  — uses trained severity head output
# =============================================================================
class DiseaseLocaliser:
    """
    Tries to load the severity-aware Faster RCNN checkpoint first.
    Falls back to the plain Faster RCNN, then to COCO-pretrained.
    Now returns: annotated_bytes, boxes, scores, severities, img_w, img_h
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self._model = None
        self._mode  = None   # "severity", "plain", or "coco"

    # ── Internal loaders ──────────────────────────────────────────────────────
    def _load_severity_model(self):
        """Load SeverityAwareFasterRCNN trained with severity head."""
        sys.path.insert(0, str(ROOT / "src"))
        from train_faster_rcnn_fast import SeverityAwareFasterRCNN
        m = SeverityAwareFasterRCNN(num_classes=NUM_CLASSES)
        ckpt = torch.load(str(RCNN_SEVERITY), map_location=self.device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        return m

    def _load_plain_model(self):
        """Load standard Faster RCNN without severity head."""
        m = fasterrcnn_resnet50_fpn(weights=None)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
        m.load_state_dict(
            torch.load(str(RCNN_PLAIN), map_location=self.device)
        )
        m.eval()
        return m

    def _ensure_loaded(self):
        if self._model is not None:
            return
        # Priority: severity checkpoint > plain checkpoint > COCO pretrained
        if RCNN_SEVERITY.exists():
            try:
                self._model = self._load_severity_model()
                self._mode  = "severity"
                print("[Localiser] Loaded severity-aware Faster RCNN")
                return
            except Exception as e:
                print(f"[Localiser] Severity model failed: {e}")
        if RCNN_PLAIN.exists():
            try:
                self._model = self._load_plain_model()
                self._mode  = "plain"
                print("[Localiser] Loaded plain Faster RCNN")
                return
            except Exception as e:
                print(f"[Localiser] Plain model failed: {e}")
        # COCO fallback
        self._mode = "coco"
        print("[Localiser] Using COCO-pretrained Faster RCNN (fallback)")

    def _run_model(self, tensor):
        """Run whichever model is loaded; return boxes, scores, and severities."""
        if self._mode == "severity":
            # SeverityAwareFasterRCNN returns detections + 'severities' key
            outputs = self._model(tensor)
            out = outputs[0]
            boxes     = out["boxes"].cpu().numpy().tolist()
            scores    = out["scores"].cpu().numpy().tolist()
            severities = out.get("severities", torch.zeros(len(boxes))).cpu().numpy().tolist()
            return boxes, scores, severities
        elif self._mode == "plain":
            out = self._model(tensor)[0]
            boxes = out["boxes"].cpu().numpy().tolist()
            scores = out["scores"].cpu().numpy().tolist()
            # Plain model has no severity head — fallback to zeros
            severities = [0.0] * len(boxes)
            return boxes, scores, severities
        else:
            # COCO fallback
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn as coco_fn,
                FasterRCNN_ResNet50_FPN_Weights,
            )
            m = coco_fn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            m.eval()
            with torch.no_grad():
                out = m(tensor)[0]
            boxes = out["boxes"].cpu().numpy().tolist()
            scores = out["scores"].cpu().numpy().tolist()
            severities = [0.0] * len(boxes)
            return boxes, scores, severities

    # ── Public detect method ──────────────────────────────────────────────────
    def detect(
        self, image_bytes: bytes, threshold: float = DETECTION_THRESHOLD
    ) -> tuple[bytes, list, list, list, int, int]:
        """
        Returns:
            annotated_bytes – JPEG with bounding boxes drawn
            boxes           – list of [x1,y1,x2,y2] above threshold
            scores          – detection scores
            severities      – learned severity per box
            width, height   – image pixel dimensions
        """
        self._ensure_loaded()

        arr  = np.frombuffer(image_bytes, np.uint8)
        bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]

        tensor = T.ToTensor()(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_boxes, raw_scores, raw_severities = self._run_model(tensor)

        # Filter by threshold
        boxes      = [b for b, s in zip(raw_boxes, raw_scores) if s >= threshold]
        scores     = [s for s in raw_scores if s >= threshold]
        severities = [sev for sev, s in zip(raw_severities, raw_scores) if s >= threshold]

        # Draw boxes with severity labels
        annotated = bgr.copy()
        for box, score, sev in zip(boxes, scores, severities):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 3)
            lbl = f"{score:.0%} | S:{sev:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+8, y1), (0, 0, 220), -1)
            cv2.putText(annotated, lbl, (x1+4, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if not boxes:
            cv2.rectangle(annotated, (4, 4), (w-4, h-4), (100, 100, 100), 2)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes(), boxes, scores, severities, w, h


LOCALISER = DiseaseLocaliser()


# =============================================================================
#  Yield Predictor
# =============================================================================
class YieldPredictor:
    def __init__(self):
        self.features = [
            "crop_encoded",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
            "disease_severity",   # ← real XGBoost input feature (now learned!)
        ]
        self.label_encoder   = LabelEncoder()
        self.yield_model     = self._train_yield_model()
        self.disease_model   = None
        self.preprocess_func = None
        self.input_size      = None
        self.class_names     = self._load_class_names()

    def _train_yield_model(self) -> XGBRegressor:
        df = pd.read_csv(DATA_DIR)
        df = df[df["Item"].isin(TARGET_CROPS)].copy()
        df["crop_encoded"]     = self.label_encoder.fit_transform(df["Item"])
        df["disease_severity"] = df["Item"].map(CROP_AVG_SEVERITY).fillna(0.0)
        clean = df[self.features + ["hg/ha_yield"]].dropna()
        X_tr, _, y_tr, _ = train_test_split(
            clean[self.features], clean["hg/ha_yield"],
            test_size=0.2, random_state=42,
        )
        m = XGBRegressor(n_estimators=100, random_state=42)
        m.fit(X_tr, y_tr)
        return m

    def _load_class_names(self) -> list[str]:
        names = sorted(p.name for p in IMAGE_DATA_DIR.iterdir() if p.is_dir())
        while len(names) < 15:
            names.append("Unknown")
        return names

    def _load_disease_model(self):
        if self.disease_model is not None:
            return
        sys.path.insert(0, str(ROOT / "src"))
        from model_factory import get_model
        model, preprocess_func, input_size = get_model(
            DISEASE_MODEL_NAME, num_classes=15, base_weights=None,
        )
        model.load_weights(CHECKPOINT)
        self.disease_model   = model
        self.preprocess_func = preprocess_func
        self.input_size      = input_size

    def predict_yield(
        self,
        crop: str, disease: str,
        rainfall: float, pesticides: float, temperature: float,
        severity: float, severity_method: str,
        pesticide_label: str | None = None,
        confidence: float | None = None,
        image_preview: str | None = None,
        n_lesion_boxes: int = 0,
        lesion_ratio: float = 0.0,
        max_severity: float = 0.0,
    ) -> YieldResult:
        """
        severity is passed as a REAL XGBoost feature.
        base_yield  = XGBoost prediction with severity=0 (healthy baseline)
        adjusted_yield = XGBoost prediction with actual severity
        """
        enc = self.label_encoder.transform([crop])[0]

        def _predict(sev):
            df = pd.DataFrame(
                [[enc, rainfall, pesticides, temperature, sev]],
                columns=self.features,
            )
            return float(self.yield_model.predict(df)[0])

        base_yield     = _predict(0.0)
        adjusted_yield = _predict(severity)

        return YieldResult(
            crop=crop, disease=disease,
            rainfall=rainfall, pesticides=pesticides,
            pesticide_label=pesticide_label or f"{fmt_number(pesticides)} tonnes",
            temperature=temperature,
            severity=severity, severity_method=severity_method,
            base_yield=base_yield, adjusted_yield=adjusted_yield,
            estimated_loss=base_yield - adjusted_yield,
            confidence=confidence, image_preview=image_preview,
            n_lesion_boxes=n_lesion_boxes, lesion_ratio=lesion_ratio,
            max_severity=max_severity,
        )

    def predict_from_image(
        self,
        image_bytes: bytes,
        rainfall: float, pesticides: float, temperature: float,
        pesticide_label: str | None = None,
    ) -> YieldResult:
        # ── Stage 2: Faster RCNN — get lesion bounding boxes + learned severities ─
        annotated_bytes, boxes, scores, severities, img_w, img_h = LOCALISER.detect(
            image_bytes, threshold=DETECTION_THRESHOLD
        )

        # ── Stage 1: EfficientNetV2S — classify disease, get CNN confidence ──
        self._load_disease_model()
        image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        resized = image.resize(self.input_size)
        arr     = np.array(resized, dtype=np.float32)
        if self.preprocess_func is not None:
            arr = self.preprocess_func(arr)
        preds      = self.disease_model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        class_idx  = int(np.argmax(preds))
        disease    = self.class_names[class_idx]
        crop       = DISEASE_TO_CROP.get(disease, "Maize")
        cnn_conf   = float(preds[class_idx])   # softmax [0,1]
        confidence = cnn_conf * 100            # percentage for display

        # ── Stage 3: Learned severity → XGBoost ──────────────────────────────
        severity, lesion_ratio, n_boxes, max_sev = compute_learned_severity(
            boxes=boxes, scores=scores, severities=severities,
            threshold=DETECTION_THRESHOLD,
        )

        preview = base64.b64encode(annotated_bytes).decode("ascii")

        return self.predict_yield(
            crop=crop, disease=disease,
            rainfall=rainfall, pesticides=pesticides, temperature=temperature,
            severity=severity, severity_method="learned",
            pesticide_label=pesticide_label,
            confidence=confidence, image_preview=preview,
            n_lesion_boxes=n_boxes, lesion_ratio=lesion_ratio,
            max_severity=max_sev,
        )


PREDICTOR: YieldPredictor | None = None

def get_predictor() -> YieldPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = YieldPredictor()
    return PREDICTOR

# def load_saved_disease_accuracy() -> float | None:
#     if not LEADERBOARD.exists():
#         return None
#     lb = pd.read_csv(LEADERBOARD)
#     m  = lb[lb["model_name"] == DISEASE_MODEL_NAME]
#     return float(m.iloc[0]["accuracy"]) if not m.empty else None

def load_saved_disease_accuracy() -> float | None:
    if not LEADERBOARD.exists():
        return None
    lb = pd.read_csv(LEADERBOARD)
    cols = {c.lower(): c for c in lb.columns}
    mcol = next((cols[c] for c in cols if "model" in c), lb.columns[0])
    acol = cols.get("accuracy") or cols.get("acc") or next((cols[c] for c in cols if "acc" in c), None)
    if acol is None:
        return None
    m = lb[lb[mcol] == DISEASE_MODEL_NAME]
    if m.empty:
        return None
    v = float(m.iloc[0][acol])
    return v / 100 if v > 1 else v      # accept 0.9335 or 93.35

def fmt_number(v: float) -> str:  return f"{v:,.0f}"
def fmt_percent(v: float | None) -> str:
    return "N/A" if v is None else f"{v*100:.1f}%"


# =============================================================================
#  HTML rendering
# =============================================================================
def render_result(result: YieldResult | None) -> str:
    if result is None:
        return ""

    if result.confidence is not None:
        conf_html = f"""
        <div class="confidence-panel">
          <span>Disease prediction confidence</span>
          <strong>{result.confidence:.1f}%</strong>
          <div class="confidence-track">
            <div class="confidence-fill"
                 style="width:{max(0,min(result.confidence,100)):.1f}%"></div>
          </div>
        </div>"""
    else:
        conf_html = """
        <div class="confidence-panel muted-panel">
          <span>Disease prediction confidence</span>
          <strong>Manual selection</strong>
          <div class="mini-note">Upload an image to get model confidence.</div>
        </div>"""

    if result.severity_method == "learned":
        badge = (
            f'<span class="tag tag-blue">Learned severity &mdash; '
            f'{result.n_lesion_boxes} lesion box'
            f'{"es" if result.n_lesion_boxes != 1 else ""}, '
            f'avg {result.severity:.3f}, max {result.max_severity:.3f}</span>'
        )
    else:
        badge = '<span class="tag tag-muted">Literature severity (manual)</span>'

    img_html = (
        f'<img class="preview" alt="Annotated leaf" '
        f'src="data:image/jpeg;base64,{result.image_preview}">'
        if result.image_preview else ""
    )

    return f"""
    <section class="result">
      <div class="section-title"><h2>Prediction Result</h2>{badge}</div>
      {img_html}
      {conf_html}
      <div class="result-grid">
        <p><span>Detected crop</span>
           <strong>{html.escape(result.crop)}</strong></p>
        <p><span>Disease</span>
           <strong>{html.escape(result.disease.replace('___',' '))}</strong></p>
        <p><span>Model accuracy</span>
           <strong>{fmt_percent(load_saved_disease_accuracy())}</strong></p>
        <p><span>Severity score</span>
           <strong>{result.severity:.4f}</strong></p>
        <p><span>Max box severity</span>
           <strong>{result.max_severity:.4f}</strong></p>
        <p><span>Severity method</span>
           <strong>{result.severity_method.capitalize()}</strong></p>
        <p><span>Rainfall</span>
           <strong>{result.rainfall:.0f} mm/year</strong></p>
        <p><span>Temperature</span>
           <strong>{result.temperature:.1f}&deg;C</strong></p>
        <p><span>Pesticide use</span>
           <strong>{html.escape(result.pesticide_label)}</strong></p>
        <p><span>Base yield (healthy)</span>
           <strong>{fmt_number(result.base_yield)} hg/ha</strong></p>
        <p><span>Predicted yield</span>
           <strong>{fmt_number(result.adjusted_yield)} hg/ha</strong></p>
        <p><span>Estimated loss</span>
           <strong>{fmt_number(result.estimated_loss)} hg/ha</strong></p>
      </div>
      <p class="severity-note">
        Severity = trained Faster R-CNN severity head output (per-box, aggregated)
        &mdash; passed directly into XGBoost as an input feature.
      </p>
    </section>"""


def render_page(result: YieldResult | None = None, error: str = "") -> bytes:
    crop_opts    = "\n".join(
        f'<option value="{html.escape(c)}">{html.escape(c)}</option>'
        for c in TARGET_CROPS)
    disease_opts = "\n".join(
        f'<option value="{html.escape(d)}">'
        f'{html.escape(d.replace("___"," "))}</option>'
        for ds in DISEASE_OPTIONS.values() for d in ds)
    pest_opts    = "\n".join(
        f'<option value="{html.escape(k)}"'
        f'{"selected" if k=="Medium" else ""}>{html.escape(k)}</option>'
        for k in PESTICIDE_LEVELS)
    acc = fmt_percent(load_saved_disease_accuracy())

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Plant Disease &amp; Yield Prediction</title>
  <style>
    :root{{--ink:#17211d;--muted:#65736d;--line:#d7e1dc;--surface:#f7faf8;
           --accent:#157a58;--dark:#0f5f44;--soft:#e4f3ec;
           --blue:#2d6cdf;--warn:#9a4d19;}}
    *{{box-sizing:border-box;}}
    body{{margin:0;font-family:Arial,sans-serif;color:var(--ink);
         background:linear-gradient(180deg,#eef5f1 0%,#f8faf9 45%,#eef3f0 100%);}}
    header{{padding:32px min(5vw,64px) 24px;background:#fff;
            border-bottom:1px solid var(--line);}}
    h1{{margin:0 0 8px;font-size:clamp(26px,4vw,42px);}}
    header p{{margin:0;color:var(--muted);max-width:820px;line-height:1.55;}}
    main{{padding:24px min(5vw,64px) 48px;}}
    .status-row{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));
                 gap:12px;margin-bottom:18px;}}
    .card{{background:#fff;border:1px solid var(--line);border-radius:8px;
           padding:14px 16px;box-shadow:0 10px 28px rgba(25,55,38,.06);}}
    .card span{{display:block;color:var(--muted);font-size:13px;margin-bottom:5px;}}
    .card strong{{font-size:20px;}}
    .layout{{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));
             gap:18px;align-items:start;}}
    form,.result{{background:#fff;border:1px solid var(--line);border-radius:8px;
                  padding:20px;box-shadow:0 14px 34px rgba(25,55,38,.07);}}
    h2{{margin:0;font-size:20px;}}
    .section-title{{display:flex;align-items:center;justify-content:space-between;
                    gap:12px;margin-bottom:14px;flex-wrap:wrap;}}
    .tag{{display:inline-flex;align-items:center;min-height:28px;padding:4px 9px;
          border-radius:999px;background:var(--soft);color:var(--dark);
          font-size:13px;font-weight:700;white-space:nowrap;}}
    .tag-blue{{background:#dce8ff;color:#1a3f7a;}}
    .tag-muted{{background:#f0f2f1;color:var(--muted);}}
    label{{display:block;margin:13px 0 6px;font-weight:700;}}
    input,select{{width:100%;min-height:42px;padding:9px 11px;
                  border:1px solid #b9c8c1;border-radius:6px;font:inherit;background:#fff;}}
    input:focus,select:focus{{outline:3px solid rgba(21,122,88,.16);border-color:var(--accent);}}
    input[type=file]{{padding:7px;}}
    .hint{{display:block;margin-top:5px;color:var(--muted);font-size:12px;}}
    button{{margin-top:16px;min-height:44px;width:100%;border:0;border-radius:6px;
            background:var(--accent);color:#fff;font-weight:700;cursor:pointer;}}
    button:hover{{background:var(--dark);}}
    .result{{margin-top:20px;}}
    .confidence-panel{{margin-bottom:14px;padding:15px;border-radius:8px;
                        border:1px solid #b8d8c9;
                        background:linear-gradient(180deg,#eef9f3 0%,#fff 100%);}}
    .confidence-panel span{{display:block;color:var(--muted);font-size:13px;margin-bottom:4px;}}
    .confidence-panel strong{{display:block;font-size:30px;margin-bottom:10px;}}
    .confidence-track{{width:100%;height:12px;border-radius:999px;background:#dce8e2;overflow:hidden;}}
    .confidence-fill{{height:100%;border-radius:inherit;
                       background:linear-gradient(90deg,var(--accent),var(--blue));}}
    .muted-panel{{border-color:var(--line);background:#f8faf9;}}
    .mini-note{{color:var(--muted);font-size:13px;line-height:1.4;}}
    .result-grid{{display:grid;grid-template-columns:repeat(3,minmax(170px,1fr));gap:12px;}}
    .result p{{margin:0;padding:13px;border:1px solid var(--line);
               border-radius:6px;background:var(--surface);}}
    .result span{{display:block;color:var(--muted);font-size:13px;margin-bottom:4px;}}
    .result strong{{font-size:18px;overflow-wrap:anywhere;}}
    .preview{{display:block;max-width:320px;width:100%;height:auto;
              margin-bottom:14px;border-radius:8px;border:1px solid var(--line);}}
    .severity-note{{margin-top:12px;font-size:13px;color:var(--muted);
                     border-top:1px solid var(--line);padding-top:10px;}}
    .error{{margin-bottom:16px;padding:12px;border-left:4px solid var(--warn);
            background:#fff7ef;color:#68310d;}}
    @media(max-width:820px){{
      .layout,.result-grid,.status-row{{grid-template-columns:1fr;}}
      .section-title{{align-items:flex-start;flex-direction:column;}}
    }}
  </style>
</head>
<body>
<header>
  <h1>Plant Disease &amp; Yield Prediction</h1>
  <p>Upload a leaf image: <strong>{DISEASE_MODEL_NAME}</strong> classifies
  the disease and <strong>Faster RCNN</strong> localises lesions.
  Severity = trained R-CNN severity head output (per-box, aggregated)
  is passed directly into XGBoost as a real input feature.</p>
</header>
<main>
  {f'<div class="error">{html.escape(error)}</div>' if error else ""}
  <section class="status-row">
    <div class="card"><span>Classifier</span><strong>{DISEASE_MODEL_NAME}</strong></div>
    <div class="card"><span>Accuracy</span><strong>{acc}</strong></div>
    <div class="card"><span>Yield model</span><strong>XGBoost</strong></div>
    <div class="card"><span>Severity</span><strong>Learned</strong></div>
  </section>
  <div class="layout">
    <form method="post" action="/predict-image" enctype="multipart/form-data">
      <div class="section-title">
        <h2>Image Detection</h2>
        <span class="tag">Learned severity</span>
      </div>
      <label for="img">Plant leaf image</label>
      <input id="img" name="image" type="file" accept=".jpg,.jpeg,.png" required>
      <span class="hint">EfficientNetV2S classifies the disease.
        Faster RCNN localises lesions + predicts per-box severity.
        Both feed XGBoost together.</span>
      <label for="r1">Rainfall (mm/year)</label>
      <input id="r1" name="rainfall" type="number" value="1000" min="0" step="any">
      <label for="t1">Temperature (&deg;C)</label>
      <input id="t1" name="temperature" type="number" value="25" step="any">
      <label for="p1">Pesticide use</label>
      <select id="p1" name="pesticide_level">{pest_opts}</select>
      <button type="submit">Detect &amp; Predict</button>
    </form>
    <form method="post" action="/predict-manual">
      <div class="section-title">
        <h2>Manual Prediction</h2>
        <span class="tag tag-muted">Literature severity</span>
      </div>
      <label for="crop">Crop</label>
      <select id="crop" name="crop">{crop_opts}</select>
      <label for="dis">Disease</label>
      <select id="dis" name="disease">{disease_opts}</select>
      <span class="hint">Uses fixed literature severity values (no image needed).</span>
      <label for="r2">Rainfall (mm/year)</label>
      <input id="r2" name="rainfall" type="number" value="1000" min="0" step="any">
      <label for="t2">Temperature (&deg;C)</label>
      <input id="t2" name="temperature" type="number" value="25" step="any">
      <label for="p2">Pesticide use</label>
      <select id="p2" name="pesticide_level">{pest_opts}</select>
      <button type="submit">Predict Yield</button>
    </form>
  </div>
  {render_result(result)}
</main>
</body>
</html>"""
    return page.encode("utf-8")


# =============================================================================
#  HTTP server
# =============================================================================
def _parse_float(fields, name, default):
    return float(fields.get(name, [str(default)])[0])

def _pest_from_level(level):
    if level not in PESTICIDE_LEVELS:
        level = "Medium"
    return PESTICIDE_LEVELS[level], level


class PredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self):   self._send(render_page())

    def do_POST(self):
        try:
            if   self.path == "/predict-manual": result = self._manual()
            elif self.path == "/predict-image":  result = self._image()
            else:
                self.send_error(404); return
            self._send(render_page(result=result))
        except Exception as exc:
            self._send(render_page(error=str(exc)))

    def _manual(self):
        body    = self.rfile.read(int(self.headers.get("content-length","0"))).decode()
        fields  = parse_qs(body)
        disease = fields.get("disease", ["Corn___Healthy"])[0]
        crop    = DISEASE_TO_CROP.get(disease, "Maize")
        pest, plabel = _pest_from_level(fields.get("pesticide_level",["Medium"])[0])
        return get_predictor().predict_yield(
            crop=crop, disease=disease,
            rainfall=_parse_float(fields,"rainfall",1000),
            pesticides=pest,
            temperature=_parse_float(fields,"temperature",25),
            severity=DISEASE_YIELD_IMPACT.get(disease, 0.0),
            severity_method="literature",
            pesticide_label=plabel,
        )

    def _image(self):
        form = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={"REQUEST_METHOD":"POST",
                     "CONTENT_TYPE":self.headers.get("content-type")},
        )
        upload = form["image"] if "image" in form else None
        if upload is None or not getattr(upload, "file", None) or not getattr(upload, "filename", ""):
            raise ValueError("Please upload an image.")
        # upload = form.get("image")
        # if upload is None or not getattr(upload,"file",None):
        #     raise ValueError("Please upload an image.")
        image_bytes = upload.file.read()
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")
        pest, plabel = _pest_from_level(form.getfirst("pesticide_level","Medium"))
        return get_predictor().predict_from_image(
            image_bytes=image_bytes,
            rainfall=float(form.getfirst("rainfall","1000")),
            pesticides=pest,
            temperature=float(form.getfirst("temperature","25")),
            pesticide_label=plabel,
        )

    def _send(self, body):
        self.send_response(200)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args): return


def main():
    host, port = "127.0.0.1", 8501
    srv = ThreadingHTTPServer((host, port), PredictionHandler)
    print(f"Interface running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    srv.serve_forever()


if __name__ == "__main__":
    main()