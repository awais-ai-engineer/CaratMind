# CaratMind Phase 6A backend

This local FastAPI backend serves the exact frozen HistGradientBoosting pipeline. It never fits a model, changes preprocessing, or evaluates the final test set. No frontend or deployment is included.

## Files and design

- `src/api/main.py`: Pydantic request/response schemas, lifespan loading, integrity checks and three endpoints.
- `src/api/input_schema.json`: training-only numeric ranges and persisted encoder categories; training-file SHA256 identifies the reference partition. No target values are included.
- `tests/test_api.py`: validation, consistency, integrity and real HTTP checks using synthetic inputs and training examples.
- `requirements-backend.txt`: exact backend, inference and test dependency versions, including transitive dependencies; TensorFlow is not needed for this backend.
- `artifacts/backend/`: verification snapshot, test reports, local server log and actual sample response. Existing phase artifacts remain untouched.

The pipeline is loaded once per application process using FastAPI's [lifespan pattern](https://fastapi.tiangolo.com/advanced/events/). Startup verifies model bytes against `artifacts/final_evaluation/frozen_selection.json` before joblib deserialization. Only trusted local artifacts should be loaded. Startup also checks model type, frozen parameters, nine original inputs, 26 transformed features, categories, inference package versions and the documented metrics file hash. Failures stop startup. The SHA256 identifies model version; it is an integrity comparison with the trusted local manifest, not a digital signature. Integrity is checked at startup, not continuously on disk.

Inference passes one ordered Pandas row directly into the persisted complete pipeline. No independent scaler or encoder is fitted or substituted. One process-wide numerical thread limit prevents excessive native thread use; synchronous prediction endpoints run in FastAPI's thread pool. One worker is the documented configuration; each additional worker would load its own pipeline.

## Environment and commands (PowerShell)

Run from `C:\Users\awais\OneDrive\Documents\CaratMind`, using Python 3.12 and the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Keep the existing model, frozen manifest, final metrics/verification files and input schema in their project-relative locations. Paths resolve from the application file rather than the shell directory. Inference-library versions are intentionally checked against the frozen training environment; an incompatible environment must not silently load the pickle. The backend does not require raw CSVs or train/validation/test CSVs at runtime. Tests do require the existing project training data and preservation snapshot.

Open `http://127.0.0.1:8000/docs` for the generated API documentation. Stop with Ctrl+C. The local server has no authentication, rate limiting or TLS termination and is not approved for public exposure. No CORS policy is needed until a frontend exists.

## Endpoints and example

`GET /health` returns operational readiness, startup artifact verification and model version; an unready model returns 503. `GET /model/info` returns identity, exact schema, training ranges, dependency versions and previously recorded validation/test performance. It does not recompute metrics.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model/info
$diamond = @{
    carat = 1.0; cut = 'Ideal'; color = 'G'; clarity = 'VS2'
    depth = 61.5; table = 57.0; x = 6.4; y = 6.4; z = 3.94
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $diamond
```

The response contains `predicted_price_usd`, `currency`, model type/version/full SHA256, warnings and an estimation disclaimer. The prediction is not rounded internally, so direct pipeline consistency is preserved. `artifacts/backend/sample_response.json` records an actual HTTP request and response.

## Validation policy

All nine fields are required. Numerical JSON values must be finite numbers; numeric strings, booleans and nulls are rejected. Integers are accepted as numerical values. No coercion, trimming, category correction or clipping is performed. Extra fields, including `price`, `source_row` and `Unnamed: 0`, return 422. Category values are case-sensitive:

- cut: Fair, Good, Ideal, Premium, Very Good
- color: D, E, F, G, H, I, J
- clarity: I1, IF, SI1, SI2, VS1, VS2, VVS1, VVS2

Carat and x/y/z must be positive. Depth/table must be in `(0,100]` because they represent percentages; this is an API domain check, not a changed preparation rule. The existing cleaning policy's y/z upper limit of 20 mm is retained. Invalid inputs return structured 422 errors without reflecting non-finite input values into JSON. An unprocessable numerical representation returns 422; a non-finite model output returns 503.

Valid values outside training min/max receive warnings, not automatic rejection. Depth disagreements exceeding 2 percentage points from `200*z/(x+y)` are retained unchanged with a warning, consistent with the original investigation. Range checks do not establish joint-distribution support; even inputs within all individual ranges may be unfamiliar combinations. Known categories are required by the API even though the existing encoder safely ignores unknown categories internally.

## Limitations

The frozen model's recorded test MAE is $271.77, RMSE $538.49, R² 0.98178 and MAPE 6.90%. These aggregate historical-dataset results are not individual error bounds. The $10,000+ test band had MAE $1,016.80 and mean prediction bias -$218.42. Extrapolation, unusual combinations, rare grades, unobserved diamond characteristics and market changes can reduce reliability. Predictions are dataset-based USD estimates, not certified appraisals or guaranteed real-time market values. No confidence intervals or individual SHAP explanations are fabricated. See `artifacts/final_evaluation/model_card.md` for the complete model card.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -q -p no:cacheprovider --junitxml=artifacts/backend/test_results.xml
```

Tests cover all endpoints, missing/extra fields, categories, nonpositive dimensions, percentage limits, wrong types, NaN/Infinity/overflowing JSON numbers, warnings, repeatability, agreement with direct persisted inference on four training observations, invalid model outputs, and startup rejection of a simulated corrupted artifact. The corruption test patches reads in memory; it never edits the artifact. An actual Uvicorn process is started on an ephemeral loopback port and stopped in `finally`. No training or test-set predictions occur. A session check compares all previous data/source/artifact hashes with the pre-implementation snapshot.
