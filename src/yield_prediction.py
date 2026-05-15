"""
Local web interface for plant disease detection and crop yield prediction.

Run from the project root:
    python src/yield_prediction.py
"""
from __future__ import annotations

import base64
import cgi
import html
import io
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "yield_data" / "yield_df.csv"
IMAGE_DATA_DIR = ROOT / "data" / "Crop___DIsease"
CHECKPOINT = ROOT / "checkpoints" / "final_ConvNeXtXLarge.weights.h5"
LEADERBOARD = ROOT / "results" / "all_models_leaderboard.csv"
DISEASE_MODEL_NAME = "ConvNeXtXLarge"

TARGET_CROPS = ["Maize", "Wheat", "Rice, paddy", "Potatoes"]

DISEASE_YIELD_IMPACT = {
    "Corn___Common_Rust": 0.35,
    "Corn___Leaf_Blight": 0.40,
    "Corn___Healthy": 0.00,
    "Potato___Early_Blight": 0.20,
    "Potato___Late_Blight": 0.45,
    "Potato___Healthy": 0.00,
    "Rice___Brown_Spot": 0.25,
    "Rice___Healthy": 0.00,
    "Rice___Hispa": 0.30,
    "Rice___Leaf_Blast": 0.50,
    "Wheat___Brown_Rust": 0.35,
    "Wheat___Healthy": 0.00,
    "Wheat___Yellow_Rust": 0.40,
    "Invalid": 0.00,
    "Unknown": 0.00,
}

DISEASE_OPTIONS = {
    "Maize": ["Corn___Common_Rust", "Corn___Leaf_Blight", "Corn___Healthy"],
    "Wheat": ["Wheat___Brown_Rust", "Wheat___Yellow_Rust", "Wheat___Healthy"],
    "Rice, paddy": [
        "Rice___Brown_Spot",
        "Rice___Hispa",
        "Rice___Leaf_Blast",
        "Rice___Healthy",
    ],
    "Potatoes": ["Potato___Early_Blight", "Potato___Late_Blight", "Potato___Healthy"],
}

DISEASE_TO_CROP = {
    "Corn___Common_Rust": "Maize",
    "Corn___Leaf_Blight": "Maize",
    "Corn___Healthy": "Maize",
    "Potato___Early_Blight": "Potatoes",
    "Potato___Late_Blight": "Potatoes",
    "Potato___Healthy": "Potatoes",
    "Rice___Brown_Spot": "Rice, paddy",
    "Rice___Hispa": "Rice, paddy",
    "Rice___Leaf_Blast": "Rice, paddy",
    "Rice___Healthy": "Rice, paddy",
    "Wheat___Brown_Rust": "Wheat",
    "Wheat___Yellow_Rust": "Wheat",
    "Wheat___Healthy": "Wheat",
    "Invalid": "Maize",
    "Unknown": "Maize",
}

PESTICIDE_LEVELS = {
    "None": 0.0,
    "Low": 1700.0,
    "Medium": 17500.0,
    "High": 49000.0,
    "Very high": 100000.0,
}


@dataclass
class YieldResult:
    crop: str
    disease: str
    rainfall: float
    pesticides: float
    pesticide_label: str
    temperature: float
    severity: float
    base_yield: float
    adjusted_yield: float
    estimated_loss: float
    confidence: float | None = None
    image_preview: str | None = None


class YieldPredictor:
    def __init__(self) -> None:
        self.features = [
            "crop_encoded",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
        ]
        self.label_encoder = LabelEncoder()
        self.yield_model = self._train_yield_model()
        self.disease_model = None
        self.preprocess_func = None
        self.input_size = None
        self.class_names = self._load_class_names()
        self.disease_model_accuracy = self._load_disease_model_accuracy()

    def _train_yield_model(self):
        df = pd.read_csv(DATA_DIR)
        df = df[df["Item"].isin(TARGET_CROPS)].copy()
        df["crop_encoded"] = self.label_encoder.fit_transform(df["Item"])

        clean = df[self.features + ["hg/ha_yield"]].dropna()
        x_train, _, y_train, _ = train_test_split(
            clean[self.features],
            clean["hg/ha_yield"],
            test_size=0.2,
            random_state=42,
        )

        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        return model

    def _load_class_names(self) -> list[str]:
        names = sorted(p.name for p in IMAGE_DATA_DIR.iterdir() if p.is_dir())
        while len(names) < 15:
            names.append("Unknown")
        return names

    def _load_disease_model_accuracy(self) -> float | None:
        if not LEADERBOARD.exists():
            return None
        leaderboard = pd.read_csv(LEADERBOARD)
        matches = leaderboard[leaderboard["model_name"] == DISEASE_MODEL_NAME]
        if matches.empty:
            return None
        return float(matches.iloc[0]["accuracy"])

    def _load_disease_model(self) -> None:
        if self.disease_model is not None:
            return

        sys.path.insert(0, str(ROOT / "src"))
        from model_factory import get_model

        model, preprocess_func, input_size = get_model(
            DISEASE_MODEL_NAME,
            num_classes=15,
            base_weights=None,
        )
        model.load_weights(CHECKPOINT)
        self.disease_model = model
        self.preprocess_func = preprocess_func
        self.input_size = input_size

    def predict_yield(
        self,
        crop: str,
        disease: str,
        rainfall: float,
        pesticides: float,
        temperature: float,
        pesticide_label: str | None = None,
        confidence: float | None = None,
        image_preview: str | None = None,
    ) -> YieldResult:
        crop_encoded = self.label_encoder.transform([crop])[0]
        input_df = pd.DataFrame(
            [[crop_encoded, rainfall, pesticides, temperature]],
            columns=self.features,
        )

        base_yield = float(self.yield_model.predict(input_df)[0])
        severity = DISEASE_YIELD_IMPACT.get(disease, 0.0)
        adjusted_yield = base_yield * (1 - severity)

        return YieldResult(
            crop=crop,
            disease=disease,
            rainfall=rainfall,
            pesticides=pesticides,
            pesticide_label=pesticide_label or f"{fmt_number(pesticides)} tonnes",
            temperature=temperature,
            severity=severity,
            base_yield=base_yield,
            adjusted_yield=adjusted_yield,
            estimated_loss=base_yield - adjusted_yield,
            confidence=confidence,
            image_preview=image_preview,
        )

    def predict_from_image(
        self,
        image_bytes: bytes,
        rainfall: float,
        pesticides: float,
        temperature: float,
        pesticide_label: str | None = None,
    ) -> YieldResult:
        self._load_disease_model()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        resized = image.resize(self.input_size)
        image_array = np.array(resized, dtype=np.float32)
        if self.preprocess_func is not None:
            image_array = self.preprocess_func(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        preds = self.disease_model.predict(image_array, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        disease = self.class_names[class_idx]
        crop = DISEASE_TO_CROP.get(disease, "Maize")
        confidence = float(preds[class_idx] * 100)
        preview = base64.b64encode(image_bytes).decode("ascii")

        return self.predict_yield(
            crop=crop,
            disease=disease,
            rainfall=rainfall,
            pesticides=pesticides,
            temperature=temperature,
            pesticide_label=pesticide_label,
            confidence=confidence,
            image_preview=preview,
        )


PREDICTOR: YieldPredictor | None = None


def get_predictor() -> YieldPredictor:
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = YieldPredictor()
    return PREDICTOR


def load_saved_disease_accuracy() -> float | None:
    if not LEADERBOARD.exists():
        return None
    leaderboard = pd.read_csv(LEADERBOARD)
    matches = leaderboard[leaderboard["model_name"] == DISEASE_MODEL_NAME]
    if matches.empty:
        return None
    return float(matches.iloc[0]["accuracy"])


def fmt_number(value: float) -> str:
    return f"{value:,.0f}"


def fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def render_result(result: YieldResult | None) -> str:
    if result is None:
        return ""

    confidence = ""
    if result.confidence is not None:
        confidence = f"""
        <div class="confidence-panel">
          <span>Disease prediction confidence</span>
          <strong>{result.confidence:.1f}%</strong>
          <div class="confidence-track">
            <div class="confidence-fill" style="width: {max(0, min(result.confidence, 100)):.1f}%"></div>
          </div>
        </div>
        """
    else:
        confidence = """
        <div class="confidence-panel muted-panel">
          <span>Disease prediction confidence</span>
          <strong>Manual selection</strong>
          <div class="mini-note">Upload an image to get model confidence for the detected class.</div>
        </div>
        """

    image = ""
    if result.image_preview:
        image = (
            '<img class="preview" alt="Uploaded plant image" '
            f'src="data:image/jpeg;base64,{result.image_preview}">'
        )

    return f"""
    <section class="result">
      <div class="section-title">
        <h2>Prediction Result</h2>
      </div>
      {image}
      {confidence}
      <div class="result-grid">
        <p><span>Detected crop</span><strong>{html.escape(result.crop)}</strong></p>
        <p><span>Disease</span><strong>{html.escape(result.disease.replace("___", " "))}</strong></p>
        <p><span>Model accuracy</span><strong>{fmt_percent(load_saved_disease_accuracy())}</strong></p>
        <p><span>Disease impact</span><strong>{result.severity * 100:.0f}%</strong></p>
        <p><span>Rainfall</span><strong>{result.rainfall:.0f} mm/year</strong></p>
        <p><span>Temperature</span><strong>{result.temperature:.1f} C</strong></p>
        <p><span>Pesticide use</span><strong>{html.escape(result.pesticide_label)}</strong></p>
        <p><span>Base yield</span><strong>{fmt_number(result.base_yield)} hg/ha</strong></p>
        <p><span>Adjusted yield</span><strong>{fmt_number(result.adjusted_yield)} hg/ha</strong></p>
        <p><span>Estimated loss</span><strong>{fmt_number(result.estimated_loss)} hg/ha</strong></p>
      </div>
    </section>
    """


def render_page(result: YieldResult | None = None, error: str = "") -> bytes:
    crop_options = "\n".join(
        f'<option value="{html.escape(crop)}">{html.escape(crop)}</option>'
        for crop in TARGET_CROPS
    )
    disease_options = "\n".join(
        f'<option value="{html.escape(disease)}">{html.escape(disease.replace("___", " "))}</option>'
        for diseases in DISEASE_OPTIONS.values()
        for disease in diseases
    )
    pesticide_options = "\n".join(
        f'<option value="{html.escape(level)}" {"selected" if level == "Medium" else ""}>{html.escape(level)}</option>'
        for level in PESTICIDE_LEVELS
    )

    accuracy_label = fmt_percent(load_saved_disease_accuracy())

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Plant Disease & Yield Prediction</title>
  <style>
    :root {{
      --ink: #17211d;
      --muted: #65736d;
      --line: #d7e1dc;
      --surface: #f7faf8;
      --accent: #157a58;
      --accent-dark: #0f5f44;
      --accent-soft: #e4f3ec;
      --blue: #2d6cdf;
      --warn: #9a4d19;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #eef5f1 0%, #f8faf9 45%, #eef3f0 100%);
    }}
    header {{
      padding: 32px min(5vw, 64px) 24px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 44px); letter-spacing: 0; }}
    header p {{ margin: 0; color: var(--muted); max-width: 820px; line-height: 1.55; }}
    main {{ padding: 24px min(5vw, 64px) 48px; }}
    .status-row {{
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .status-card {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 10px 28px rgba(25, 55, 38, 0.06);
    }}
    .status-card span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 5px;
    }}
    .status-card strong {{
      font-size: 22px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    form, .result {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 14px 34px rgba(25, 55, 38, 0.07);
    }}
    h2 {{ margin: 0; font-size: 20px; }}
    .section-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .tag {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }}
    label {{ display: block; margin: 13px 0 6px; font-weight: 700; }}
    input, select {{
      width: 100%;
      min-height: 42px;
      padding: 9px 11px;
      border: 1px solid #b9c8c1;
      border-radius: 6px;
      font: inherit;
      background: #fff;
    }}
    input:focus, select:focus {{
      outline: 3px solid rgba(21, 122, 88, 0.16);
      border-color: var(--accent);
    }}
    input[type=file] {{ padding: 7px; }}
    .hint {{
      display: block;
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    button {{
      margin-top: 16px;
      min-height: 44px;
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    .result {{ margin-top: 20px; }}
    .confidence-panel {{
      margin-bottom: 14px;
      padding: 15px;
      border-radius: 8px;
      border: 1px solid #b8d8c9;
      background: linear-gradient(180deg, #eef9f3 0%, #ffffff 100%);
    }}
    .confidence-panel span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .confidence-panel strong {{
      display: block;
      font-size: 30px;
      margin-bottom: 10px;
    }}
    .confidence-track {{
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: #dce8e2;
      overflow: hidden;
    }}
    .confidence-fill {{
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--blue));
    }}
    .muted-panel {{
      border-color: var(--line);
      background: #f8faf9;
    }}
    .mini-note {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .result-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 12px;
    }}
    .result p {{
      margin: 0;
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
    }}
    .result span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
    .result strong {{ font-size: 18px; overflow-wrap: anywhere; }}
    .preview {{
      display: block;
      max-width: 320px;
      width: 100%;
      height: auto;
      margin-bottom: 14px;
      border-radius: 8px;
      border: 1px solid var(--line);
    }}
    .error {{
      margin-bottom: 16px;
      padding: 12px;
      border-left: 4px solid var(--warn);
      background: #fff7ef;
      color: #68310d;
    }}
    @media (max-width: 820px) {{
      .layout, .result-grid, .status-row {{ grid-template-columns: 1fr; }}
      .section-title {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Plant Disease & Yield Prediction</h1>
    <p>Upload a leaf image to detect the disease, or choose the crop and disease manually. Rainfall, temperature, and pesticides are entered manually and used for yield prediction.</p>
  </header>
  <main>
    {f'<div class="error">{html.escape(error)}</div>' if error else ''}
    <section class="status-row">
      <div class="status-card"><span>Disease model</span><strong>{DISEASE_MODEL_NAME}</strong></div>
      <div class="status-card"><span>Saved accuracy</span><strong>{accuracy_label}</strong></div>
      <div class="status-card"><span>Yield model</span><strong>XGBoost</strong></div>
    </section>
    <div class="layout">
      <form method="post" action="/predict-image" enctype="multipart/form-data">
        <div class="section-title">
          <h2>Image Detection</h2>
          <span class="tag">Class + yield</span>
        </div>
        <label for="image">Plant image</label>
        <input id="image" name="image" type="file" accept=".jpg,.jpeg,.png" required>
        <label for="rainfall-image">Rainfall (mm/year)</label>
        <input id="rainfall-image" name="rainfall" type="number" value="1000" min="0" step="any">
        <label for="temp-image">Temperature (C)</label>
        <input id="temp-image" name="temperature" type="number" value="25" step="any">
        <label for="pesticides-image">Pesticide use</label>
        <select id="pesticides-image" name="pesticide_level">{pesticide_options}</select>
        <span class="hint">Choose the closest farm-level use level.</span>
        <button type="submit">Detect & Predict</button>
      </form>

      <form method="post" action="/predict-manual">
        <div class="section-title">
          <h2>Manual Prediction</h2>
          <span class="tag">Yield only</span>
        </div>
        <label for="crop">Crop</label>
        <select id="crop" name="crop">{crop_options}</select>
        <label for="disease">Disease</label>
        <select id="disease" name="disease">{disease_options}</select>
        <label for="rainfall-manual">Rainfall (mm/year)</label>
        <input id="rainfall-manual" name="rainfall" type="number" value="1000" min="0" step="any">
        <label for="temp-manual">Temperature (C)</label>
        <input id="temp-manual" name="temperature" type="number" value="25" step="any">
        <label for="pesticides-manual">Pesticide use</label>
        <select id="pesticides-manual" name="pesticide_level">{pesticide_options}</select>
        <span class="hint">Choose the closest farm-level use level.</span>
        <button type="submit">Predict Yield</button>
      </form>
    </div>
    {render_result(result)}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


def parse_float(fields: dict[str, list[str]], name: str, default: float) -> float:
    value = fields.get(name, [str(default)])[0]
    return float(value)


def pesticide_from_level(level: str) -> tuple[float, str]:
    if level not in PESTICIDE_LEVELS:
        level = "Medium"
    return PESTICIDE_LEVELS[level], level


class PredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send(render_page())

    def do_POST(self) -> None:
        try:
            if self.path == "/predict-manual":
                result = self._handle_manual_prediction()
            elif self.path == "/predict-image":
                result = self._handle_image_prediction()
            else:
                self.send_error(404)
                return
            self._send(render_page(result=result))
        except Exception as exc:
            self._send(render_page(error=str(exc)))

    def _handle_manual_prediction(self) -> YieldResult:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        fields = parse_qs(body)

        crop = fields.get("crop", ["Maize"])[0]
        disease = fields.get("disease", ["Corn___Healthy"])[0]
        if DISEASE_TO_CROP.get(disease) in TARGET_CROPS:
            crop = DISEASE_TO_CROP[disease]
        pesticides, pesticide_label = pesticide_from_level(
            fields.get("pesticide_level", ["Medium"])[0]
        )

        return get_predictor().predict_yield(
            crop=crop,
            disease=disease,
            rainfall=parse_float(fields, "rainfall", 1000),
            pesticides=pesticides,
            temperature=parse_float(fields, "temperature", 25),
            pesticide_label=pesticide_label,
        )

    def _handle_image_prediction(self) -> YieldResult:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("content-type"),
            },
        )

        upload = form["image"] if "image" in form else None
        if upload is None or not getattr(upload, "file", None):
            raise ValueError("Please upload an image.")

        image_bytes = upload.file.read()
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")

        pesticides, pesticide_label = pesticide_from_level(
            form.getfirst("pesticide_level", "Medium")
        )

        return get_predictor().predict_from_image(
            image_bytes=image_bytes,
            rainfall=float(form.getfirst("rainfall", "1000")),
            pesticides=pesticides,
            temperature=float(form.getfirst("temperature", "25")),
            pesticide_label=pesticide_label,
        )

    def _send(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    host = "127.0.0.1"
    port = 8501
    server = ThreadingHTTPServer((host, port), PredictionHandler)
    print(f"Interface running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
