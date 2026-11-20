"""Serve the unchanged, manifest-verified production pipeline."""
from contextlib import asynccontextmanager
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

# Match the selected ANN's deterministic CPU inference configuration.
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
DISCLAIMER = "Dataset-based USD price estimate, not a certified appraisal or guaranteed current market valuation. No individual confidence interval is provided."
logger = logging.getLogger(__name__)


class DiamondInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    carat: float = Field(gt=0, description="Diamond weight in carats")
    cut: Literal["Fair", "Good", "Ideal", "Premium", "Very Good"]
    color: Literal["D", "E", "F", "G", "H", "I", "J"]
    clarity: Literal["I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"]
    depth: float = Field(gt=0, le=100, description="Total depth percentage")
    table: float = Field(gt=0, le=100, description="Table percentage")
    x: float = Field(gt=0, description="Length in mm")
    y: float = Field(gt=0, le=20, description="Width in mm; existing preparation limit")
    z: float = Field(gt=0, le=20, description="Depth in mm; existing preparation limit")


class PredictionResponse(BaseModel):
    predicted_price_usd: float = Field(allow_inf_nan=False)
    currency: Literal["USD"] = "USD"
    model_type: str
    model_version: str
    model_sha256: str
    warnings: list[str]
    disclaimer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    manifest = json.loads((ROOT / "artifacts/final_evaluation/frozen_selection.json").read_text())
    if manifest["primary_model"] != "HistGradientBoostingRegressor" or manifest["input_columns"] != FEATURES:
        raise RuntimeError("Unexpected frozen model or input schema")
    relative_path = manifest["model_paths"][manifest["primary_model"]]
    artifact = (ROOT / relative_path).read_bytes()
    model_hash = hashlib.sha256(artifact).hexdigest()
    expected_hash = manifest["protected_sha256"][str(Path(relative_path))]
    if model_hash != expected_hash:
        raise RuntimeError("Frozen model SHA256 mismatch; refusing to load")
    # Fail closed rather than unpickle a model under different inference libraries.
    libraries = ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]
    for package in libraries:
        if version(package) != manifest["packages"][package]:
            raise RuntimeError(f"Inference package version mismatch: {package}")
    pipeline = joblib.load(BytesIO(artifact))
    estimator = pipeline.named_steps["model"]
    preprocessing = pipeline.named_steps["preprocessing"]
    if not isinstance(estimator, HistGradientBoostingRegressor):
        raise RuntimeError("Unexpected model class")
    if estimator.get_params() != manifest["model_parameters"][manifest["primary_model"]]:
        raise RuntimeError("Model configuration differs from frozen selection")
    if pipeline.feature_names_in_.tolist() != FEATURES or estimator.n_features_in_ != 26:
        raise RuntimeError("Pipeline input dimensions differ from training")
    if preprocessing.get_feature_names_out().tolist() != manifest["transformed_features"]:
        raise RuntimeError("Transformed feature order differs from training")
    schema = json.loads((Path(__file__).parent / "input_schema.json").read_text())
    if schema["training_sha256"] != manifest["protected_sha256"][str(Path("data/processed/train.csv"))]:
        raise RuntimeError("Range reference does not identify the frozen training partition")
    encoder = preprocessing.named_steps["columns"].named_transformers_["categorical"]
    for name, categories in zip(["cut", "color", "clarity"], encoder.categories_):
        declared = DiamondInput.model_fields[name].annotation.__args__
        if list(declared) != categories.tolist() or schema["categories"][name] != categories.tolist():
            raise RuntimeError(f"Category schema mismatch: {name}")
    metrics_path = ROOT / "artifacts/final_evaluation/test_metrics.json"
    verification = json.loads((ROOT / "artifacts/final_evaluation/verification.json").read_text())
    if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != verification["output_sha256"][str(Path("artifacts/final_evaluation/test_metrics.json"))]:
        raise RuntimeError("Documented performance file integrity failure")
    metrics = next(row for row in json.loads(metrics_path.read_text()) if row["model"] == manifest["primary_model"])
    all_metrics = json.loads(metrics_path.read_text())
    ann_path = manifest["model_paths"]["phase4a_deep_raw"]
    ann_bytes = (ROOT / ann_path).read_bytes()
    ann_hash = hashlib.sha256(ann_bytes).hexdigest()
    if ann_hash != manifest["protected_sha256"][str(Path(ann_path))]:
        raise RuntimeError("Frozen ANN SHA256 mismatch")
    preprocessing_path = "artifacts/preparation/preprocessor.joblib"
    preprocessing_bytes = (ROOT / preprocessing_path).read_bytes()
    if hashlib.sha256(preprocessing_bytes).hexdigest() != manifest["protected_sha256"][str(Path(preprocessing_path))]:
        raise RuntimeError("Frozen ANN preprocessing SHA256 mismatch")
    for package in ["tensorflow", "keras"]:
        if version(package) != manifest["packages"][package]:
            raise RuntimeError(f"ANN package version mismatch: {package}")
    import tensorflow as tf
    from tensorflow import keras
    # TensorFlow permits setting thread counts only before its first execution.
    if not getattr(app.state, "tensorflow_configured", False):
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        app.state.tensorflow_configured = True
    ann = keras.models.load_model(ROOT / ann_path, compile=False)
    # Check again after loading: the deserialized file must still be the frozen file.
    if hashlib.sha256((ROOT / ann_path).read_bytes()).hexdigest() != ann_hash:
        raise RuntimeError("ANN artifact changed during startup")
    ann_preprocessor = joblib.load(BytesIO(preprocessing_bytes))
    if ann.input_shape != (None, 26) or ann.output_shape != (None, 1) or ann.count_params() != 13825:
        raise RuntimeError("Selected ANN architecture mismatch")
    if ann_preprocessor.feature_names_in_.tolist() != FEATURES or ann_preprocessor.get_feature_names_out().tolist() != manifest["transformed_features"]:
        raise RuntimeError("ANN preprocessing feature order mismatch")
    app.state.ann = ann
    app.state.ann_preprocessor = ann_preprocessor
    app.state.ann_hash = ann_hash
    app.state.ann_version = f"phase4a-seed42-{ann_hash[:12]}"
    app.state.ann_info = {"model_type": "Deep ANN", "model_version": app.state.ann_version,
        "model_sha256": ann_hash, "seed": 42, "architecture": [128, 64, 32, 1],
        "input_features": FEATURES, "transformed_feature_count": 26, "artifact_verified": True,
        "test_performance": next(row for row in all_metrics if row["model"] == "phase4a_deep_raw"),
        "disclaimer": DISCLAIMER}
    importance_path = "artifacts/final_evaluation/permutation_importance.csv"
    if hashlib.sha256((ROOT / importance_path).read_bytes()).hexdigest() != verification["output_sha256"][str(Path(importance_path))]:
        raise RuntimeError("Feature importance report integrity failure")
    app.state.insights = {"importance": pd.read_csv(ROOT / importance_path).to_dict("records"),
        "reference": "Validation set, five permutations, seed 42; increase in MAE (USD).",
        "limitations": "Predictive reliance, not causation. Correlated carat/x/y/z can share or mask importance; permutations may create implausible combinations."}
    app.state.pipeline = pipeline
    app.state.schema = schema
    app.state.model_hash = model_hash
    app.state.model_version = f"phase3b-{model_hash[:12]}"
    app.state.info = {
        "model_type": manifest["primary_model"], "model_version": app.state.model_version,
        "model_sha256": model_hash, "artifact_verified": True,
        "input_features": FEATURES, "transformed_feature_count": 26,
        "categories": schema["categories"], "training_numeric_ranges": schema["numeric_ranges"],
        "test_performance": metrics, "inference_packages": {p: version(p) for p in libraries},
        "disclaimer": DISCLAIMER,
        "comparison_models": all_metrics,
    }
    logger.info("Loaded frozen pipeline once: %s", model_hash)
    # Apply one process-wide thread limit, rather than changing global limits per request.
    with threadpool_limits(limits=1):
        yield
    app.state.pipeline = None
    app.state.ann = None


app = FastAPI(title="CaratMind Prediction API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    # Avoid reflecting NaN/Infinity into a JSON response (JSON cannot encode them).
    details = [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health")
def health(request: Request):
    if getattr(request.app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ok", "model_loaded": True, "artifact_verified": True,
            "model_version": request.app.state.model_version}


@app.get("/model/info")
def model_info(request: Request):
    health(request)
    return request.app.state.info


@app.get("/model/info/ann")
def ann_info(request: Request):
    health(request)
    return request.app.state.ann_info


@app.get("/model/insights")
def model_insights(request: Request):
    health(request)
    return request.app.state.insights


def input_warnings(diamond: DiamondInput, request: Request):
    values = diamond.model_dump()
    warnings = []
    for feature, bounds in request.app.state.schema["numeric_ranges"].items():
        if not bounds["min"] <= values[feature] <= bounds["max"]:
            warnings.append(f"{feature} is outside the training range [{bounds['min']}, {bounds['max']}]; extrapolation reliability is unknown.")
    calculated_depth = 200 * (values["z"] / (values["x"] + values["y"]))
    if abs(values["depth"] - calculated_depth) > 2:
        warnings.append("Depth differs by more than 2 percentage points from 200*z/(x+y); verify measurements. Input was retained unchanged.")
    return warnings


@app.post("/predict", response_model=PredictionResponse)
def predict(diamond: DiamondInput, request: Request):
    health(request)
    warnings = input_warnings(diamond, request)
    frame = pd.DataFrame([diamond.model_dump()], columns=FEATURES)
    try:
        prediction = float(request.app.state.pipeline.predict(frame)[0])
    except (ValueError, OverflowError) as error:
        logger.warning("Input could not be represented by the frozen pipeline: %s", error)
        raise HTTPException(status_code=422, detail="Measurements cannot be processed safely by the frozen pipeline") from error
    if not np.isfinite(prediction):
        logger.error("Frozen pipeline returned a non-finite prediction")
        raise HTTPException(status_code=503, detail="Model returned an invalid prediction")
    return PredictionResponse(predicted_price_usd=prediction, model_type="HistGradientBoostingRegressor",
        model_version=request.app.state.model_version, model_sha256=request.app.state.model_hash,
        warnings=warnings, disclaimer=DISCLAIMER)


@app.post("/predict/ann", response_model=PredictionResponse)
def predict_ann(diamond: DiamondInput, request: Request):
    health(request)
    if getattr(request.app.state, "ann", None) is None:
        raise HTTPException(status_code=503, detail="ANN is not ready")
    frame = pd.DataFrame([diamond.model_dump()], columns=FEATURES)
    try:
        transformed = request.app.state.ann_preprocessor.transform(frame).astype("float32")
        if transformed.shape != (1, 26) or not np.isfinite(transformed).all():
            raise ValueError("ANN input cannot be represented as 26 finite float32 values")
        prediction = float(request.app.state.ann(transformed, training=False).numpy().ravel()[0])
    except (ValueError, OverflowError) as error:
        raise HTTPException(status_code=422, detail="Measurements cannot be processed safely by the ANN") from error
    if not np.isfinite(prediction):
        raise HTTPException(status_code=503, detail="ANN returned an invalid prediction")
    return PredictionResponse(predicted_price_usd=prediction, model_type="Deep ANN",
        model_version=request.app.state.ann_version, model_sha256=request.app.state.ann_hash,
        warnings=input_warnings(diamond, request), disclaimer=DISCLAIMER)
