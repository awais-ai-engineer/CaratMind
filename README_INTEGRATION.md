# CaratMind Phase 6B: ML, Deep Learning and frontend

This phase adds an independent ANN endpoint and a Next.js/TypeScript interface. The frozen primary estimator remains tuned HistGradientBoosting. No model fitting, retuning, ensemble averaging, test-set evaluation or deployment occurs.

## Structure

- `src/api/main.py`: existing FastAPI application, extended with ANN startup verification, independent inference and read-only reporting endpoints. Shared request validation and input warnings serve both models.
- `src/api/input_schema.json`: unchanged training-only ranges and category schema.
- `frontend/app/page.tsx`: straightforward React state for verified metadata, form values, loading/errors and separate model responses. No prediction calculations occur in the browser; only formatting and the absolute difference between returned estimates.
- `frontend/app/globals.css`: responsive design tokens, layout, forms, charts and keyboard focus styles.
- `frontend/app/api/[...path]/route.ts`: same-origin `/api/*` proxy with a fixed endpoint/method allowlist and a timeout. `API_BASE_URL` is read at runtime; browsers never need a hardcoded backend host or cross-origin access.
- `frontend/tests/app.spec.ts`: real browser integration plus explicitly isolated failure injection for loading and error states.
- `tests/test_ann_api.py`: saved-model parity, validation, warnings, identity and no-fitting checks. Existing 70 regression tests remain in `tests/test_api.py`; its live-server outputs now use the integration artifact directory so Phase 6A results are preserved.
- `artifacts/integration/`: pre-change hashes/backups, test reports, screenshots, actual responses and verification results.

## Environment

Use the existing Python 3.12 `.venv`, Node.js and npm. Exact Python versions are pinned in `requirements-backend.txt`, including the already installed TensorFlow/Keras versions. Frontend direct dependencies are exact versions in `frontend/package.json`; `package-lock.json` pins the complete resolved tree. Use `npm ci` for repeat installs. Existing ML/ANN inference library versions must match the frozen manifest. Nothing in this phase upgrades the scientific Python environment.

### Terminal 1: FastAPI

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

TensorFlow CPU startup can take noticeably longer than the ML-only backend. Startup failures stop readiness rather than silently omitting the ANN. The model loads once per process; do not use `--reload` when measuring startup or inference behavior. Stop with Ctrl+C.

### Terminal 2: Next.js

```powershell
cd frontend
Copy-Item .env.example .env.local
npm ci
npm run dev
```

Open `http://127.0.0.1:3000`. `API_BASE_URL` in `.env.local` configures the backend origin. Restart Next.js after changing it. Its default is loopback port 8000 for local development; no production host is embedded. The application needs a running Next.js server for proxying and is not a static export. No public deployment, authentication, TLS or rate limiting is provided.

## Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Backend readiness after both model loads |
| GET | `/model/info` | Existing primary metadata, verified saved comparison metrics |
| GET | `/model/info/ann` | Selected seed-42 ANN identity and saved performance |
| GET | `/model/insights` | Hash-verified validation permutation importance |
| POST | `/predict` | Unchanged primary boosting estimate |
| POST | `/predict/ann` | Independent ANN estimate in original USD |

Both prediction endpoints require exactly nine original features. Category names are case-sensitive; finite positive numerical inputs are required. Existing measurement constraints and out-of-training-range warnings are shared. Invalid requests return 422. Non-finite model predictions return 503. No user input is clipped, repaired or averaged. Readiness and inference failures are displayed in the UI; if only one inference request fails, the successful independent result is retained.

```powershell
$diamond = @{
    carat = 1.0; cut = 'Ideal'; color = 'G'; clarity = 'VS2'
    depth = 61.5; table = 57.0; x = 6.4; y = 6.4; z = 3.94
} | ConvertTo-Json
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $diamond
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict/ann -ContentType 'application/json' -Body $diamond
```

Prediction responses include USD price, model type, full SHA256, version, training-domain warnings and the estimation disclaimer. No individual SHAP value or confidence interval is invented.

## Artifact integrity and scientific interpretation

The original frozen selection manifest supplies both model identities. Startup verifies the selected ANN SHA256 and standalone preprocessing SHA256, TensorFlow/Keras versions, exactly 26 transformed inputs, original feature order, output shape and parameter count (13,825). The original full ML pipeline remains intact. ANN inference uses the original fitted preprocessor, float32 inputs, `training=False` and original-price output. The saved `.keras` file is loaded without compilation; no optimizer or training code is needed. Trusted local manifests and artifacts are required; hashes are not signatures.

All model performance comes from the existing hash-verified final metrics report, including the earlier validation metrics stored in it. The frontend does not rerun evaluation. Feature importance is read from the hash-verified permutation report: validation labels, five repeats, original feature groups. It describes predictive reliance, not causation. Correlated dimensions and unrealistic perturbed combinations limit interpretation.

The ANN is a separate demonstration of Deep Learning, not a replacement for the validation-selected production candidate. The absolute price difference represents model disagreement, not uncertainty or an error bound. Aggregate test metrics do not guarantee individual accuracy. Outputs are historical dataset-based estimates, not certified appraisals or guaranteed current market valuations. Unfamiliar inputs, rare grades, expensive diamonds and market changes need caution. See the unchanged `artifacts/final_evaluation/model_card.md`.

## Verification commands

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_ann_api.py -q -p no:cacheprovider --junitxml=artifacts/integration/backend_tests.xml
```

From `frontend`:

```powershell
npm run typecheck
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests start temporary backend/frontend processes on loopback ports 8100/3100, use the real models, then stop the servers. Keep these ports free. They exercise desktop (1440px) and mobile (390px), real proxy requests, metadata/health, valid predictions, invalid input, training-range warnings, loading states, partial failures, backend unavailability, keyboard navigation, overflow and axe accessibility checks. Error-state mocks deliberately return errors only; they never fabricate model estimates. Screenshot evidence and test reports are local artifacts, not proof of full cross-browser, assistive-technology or load-testing coverage.

To use an already installed Google Chrome instead of downloading Chromium, run `$env:PLAYWRIGHT_CHANNEL = 'chrome'` before `npm run test:e2e`. This uses a separate temporary browser context and does not operate personal tabs.

Before deployment, separately review authentication/access policy, TLS, request limits, dependency security, monitoring, resource limits, operational testing and model-governance requirements. No deployment is authorized or performed in this phase.

## Recorded Phase 6B results

- Backend: 83 tests passed in the combined run, plus one additional ANN corruption-startup test passed; 84 passing checks total, including all original 70 assertions.
- Browser: final run passed all 12 tests (six desktop, six mobile). Both axe audits reported zero violations in the selected WCAG rule sets. This is automated coverage, not a complete accessibility certification.
- TypeScript and optimized Next.js build passed. `pip check` passed; npm installation audit reported zero known vulnerabilities.
- Actual example response: boosting $6,136.93; ANN $6,298.10; absolute difference $161.17. Both models returned their verified identities and no domain warnings for the supplied example. Full JSON and screenshots are in `artifacts/integration/`.
- Initial browser run: four passed, eight failed because broad alert selectors also matched Next.js's route announcer and an exact label lookup missed the select control. Failure reports were preserved; scoped alert selectors and a role-based combobox selector corrected the tests. No inference logic or frozen artifact was changed to resolve these failures.
- Warnings retained: Starlette's HTTPX test-client deprecation; expected native-Windows TensorFlow CPU/GPU notices; harmless Node terminal-color environment warning.
- Verification covers Windows, installed Chrome at 1440x1000 and a 390x844 mobile viewport. Real mobile hardware, Safari/Firefox, assistive technology, concurrency/load tests and public deployment remain outside this phase.
