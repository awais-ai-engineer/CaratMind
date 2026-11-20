"use client";

import { FormEvent, useEffect, useState } from "react";

type Metrics = {
  model: string;
  mae_usd: number;
  rmse_usd: number;
  r2: number;
  mape_percent: number;
  validation_mae_usd: number;
  validation_rmse_usd: number;
  validation_r2: number;
  sample_count: number;
};
type ModelInfo = {
  model_type: string;
  model_version: string;
  artifact_verified: boolean;
  categories: Record<string, string[]>;
  training_numeric_ranges: Record<string, { min: number; max: number }>;
  test_performance: Metrics;
  comparison_models: Metrics[];
};
type Prediction = {
  predicted_price_usd: number;
  model_type: string;
  model_version: string;
  warnings: string[];
  disclaimer: string;
};
type Insight = {
  importance: {
    feature: string;
    mean_mae_increase_usd: number;
    repeat_sd_usd: number;
  }[];
  reference: string;
  limitations: string;
};
type Health = { status: string; artifact_verified: boolean };
const numericFields = ["carat", "depth", "table", "x", "y", "z"];
const fieldLabels: Record<string, string> = {
  carat: "Carat weight",
  depth: "Depth (%)",
  table: "Table (%)",
  x: "Length x (mm)",
  y: "Width y (mm)",
  z: "Height z (mm)",
};
const defaults: Record<string, string> = {
  carat: "1.0",
  cut: "Ideal",
  color: "G",
  clarity: "VS2",
  depth: "61.5",
  table: "57",
  x: "6.4",
  y: "6.4",
  z: "3.94",
};
const usd = (value: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
const modelName = (name: string) =>
  name === "phase4a_deep_raw"
    ? "Deep ANN · seed 42"
    : name === "RandomForestRegressor"
      ? "Random Forest"
      : "HistGradientBoosting";

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const timeout = AbortSignal.timeout(30000);
  const response = await fetch(`/api${path}`, {
    ...options,
    signal: options.signal
      ? AbortSignal.any([options.signal, timeout])
      : timeout,
  });
  if (!response.ok) {
    let message = `Request failed (${response.status}). Please check that the backend is running.`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail))
        message = body.detail
          .map(
            (item: { loc?: string[]; msg?: string }) =>
              `${item.loc?.slice(1).join(".") || "Input"}: ${item.msg}`,
          )
          .join("; ");
    } catch {
      /* A proxy may return a non-JSON error page. */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function Diamond({ large = false }: { large?: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 200 160"
      className={large ? "diamond-art" : "brand-icon"}
      fill="none"
    >
      <path
        d="M40 18h120l32 44-92 90L8 62z"
        fill={large ? "#e1f0ef" : "#d2eee8"}
        stroke="currentColor"
        strokeWidth="3"
      />
      <path
        d="m40 18 30 44 30-44 30 44 30-44M8 62h184M70 62l30 90 30-90"
        stroke="currentColor"
        strokeWidth="2"
      />
      {large && (
        <>
          <path d="m8 62 92 90-30-90Z" fill="#a6d4d0" opacity=".6" />
          <path d="m130 62-30 90 92-90Z" fill="#5e9f9e" opacity=".25" />
        </>
      )}
    </svg>
  );
}

export default function Home() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [annInfo, setAnnInfo] = useState<{ model_version: string } | null>(
    null,
  );
  const [insight, setInsight] = useState<Insight | null>(null);
  const [healthy, setHealthy] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [values, setValues] = useState(defaults);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<{ ml?: Prediction; ann?: Prediction }>(
    {},
  );
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api<Health>("/health", { signal: controller.signal }),
      api<ModelInfo>("/model/info", { signal: controller.signal }),
      api<{ model_version: string }>("/model/info/ann", {
        signal: controller.signal,
      }),
      api<Insight>("/model/insights", { signal: controller.signal }),
    ])
      .then(([health, model, ann, insights]) => {
        setHealthy(health.status === "ok" && health.artifact_verified);
        setInfo(model);
        setAnnInfo(ann);
        setInsight(insights);
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setConnectionError(`Backend unavailable. ${error.message}`);
      });
    return () => controller.abort();
  }, []);

  function change(name: string, value: string) {
    setValues((previous) => ({ ...previous, [name]: value }));
    setResults({});
    setErrors([]);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const payload: Record<string, string | number> = { ...values };
    for (const field of numericFields) {
      if (
        !values[field].trim() ||
        !Number.isFinite(Number(values[field])) ||
        Number(values[field]) <= 0
      ) {
        setErrors([`${fieldLabels[field]} must be a finite positive number.`]);
        return;
      }
      payload[field] = Number(values[field]);
    }
    setBusy(true);
    setErrors([]);
    setResults({});
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    };
    const [ml, ann] = await Promise.allSettled([
      api<Prediction>("/predict", options),
      api<Prediction>("/predict/ann", options),
    ]);
    setResults({
      ml: ml.status === "fulfilled" ? ml.value : undefined,
      ann: ann.status === "fulfilled" ? ann.value : undefined,
    });
    const failures: string[] = [];
    if (ml.status === "rejected")
      failures.push(`Primary estimator: ${ml.reason.message}`);
    if (ann.status === "rejected")
      failures.push(`ANN estimator: ${ann.reason.message}`);
    setErrors(failures);
    setBusy(false);
  }
  const warnings = [
    ...new Set([
      ...(results.ml?.warnings || []),
      ...(results.ann?.warnings || []),
    ]),
  ];
  const performance = info?.test_performance;

  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <aside className="sidebar">
        <a className="brand" href="#overview">
          <Diamond />
          <span>
            CaratMind<span className="brand-sub">DIAMOND INTELLIGENCE</span>
          </span>
        </a>
        <p className="nav-label">WORKSPACE</p>
        <nav aria-label="Main navigation">
          {[
            ["overview", "01", "Overview"],
            ["prediction", "02", "Price prediction"],
            ["comparison", "03", "ML vs DL"],
            ["performance", "04", "Model performance"],
            ["insights", "05", "Model insights"],
            ["methodology", "06", "Methodology"],
          ].map(([id, number, label]) => (
            <a key={id} href={`#${id}`}>
              <span>{number}</span>
              {label}
            </a>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="tiny-label">BUILT ON EVIDENCE</span>
          <p>
            Two independent engines.
            <br />
            One transparent perspective.
          </p>
          <span className="outline-pill">AI Engineering Assessment</span>
        </div>
      </aside>
      <main id="main">
        <header className="topbar">
          <span>
            WORKSPACE <span className="slash">/</span> Diamond intelligence
          </span>
          <span
            role="status"
            className={`connection ${healthy ? "online" : ""}`}
          >
            <i />
            {healthy
              ? "Models connected"
              : connectionError
                ? "Backend offline"
                : "Connecting to backend…"}
          </span>
        </header>
        <div className="content">
          {connectionError && (
            <div role="alert" className="error-box">
              {connectionError} Start the backend, then reload this page.
            </div>
          )}
          <section id="overview" className="hero">
            <div>
              <p className="eyebrow">PRECISION MEETS PERSPECTIVE</p>
              <h1>
                A clearer view of
                <br />
                diamond value.
              </h1>
              <p className="hero-copy">
                Explore the characteristics behind every estimate.
                <br className="desktop-break" /> Compare machine learning with
                an independent deep learning perspective.
              </p>
              <a className="text-link" href="#prediction">
                Explore your diamond <span>↗</span>
              </a>
            </div>
            <div className="hero-visual">
              <Diamond large />
              <span className="art-label">DATA. DETAIL. DISCOVERY.</span>
            </div>
          </section>
          <div className="section-heading">
            <h2>Evidence at a glance</h2>
            <span className="muted small">
              Frozen primary model · independent test set
            </span>
          </div>
          <div className="metric-grid">
            {[
              [
                "TEST MAE",
                performance ? usd(performance.mae_usd) : "—",
                "Mean absolute error · USD",
              ],
              [
                "TEST RMSE",
                performance ? usd(performance.rmse_usd) : "—",
                "Root mean squared error · USD",
              ],
              [
                "TEST R²",
                performance ? performance.r2.toFixed(5) : "—",
                "Coefficient of determination",
              ],
              [
                "TEST MAPE",
                performance ? `${performance.mape_percent.toFixed(2)}%` : "—",
                "Mean absolute percentage error",
              ],
            ].map(([label, value, caption]) => (
              <article className="metric-card" key={label}>
                <span className="tiny-label">{label}</span>
                <strong>{value}</strong>
                <span>{caption}</span>
              </article>
            ))}
          </div>
          <section id="prediction" className="prediction-section">
            <div className="section-heading">
              <div>
                <p className="eyebrow">THE ESTIMATION STUDIO</p>
                <h2>Your diamond, in detail.</h2>
              </div>
              <span className="outline-pill">9 characteristics · 2 models</span>
            </div>
            <div className="studio-grid">
              <form onSubmit={submit} className="panel input-panel">
                <div className="panel-title">
                  <h3>Diamond specifications</h3>
                  <span className="muted small">Example values supplied</span>
                </div>
                <fieldset disabled={busy || !healthy}>
                  <legend className="sr-only">Diamond specifications</legend>
                  <div className="form-grid">
                    <label className="field">
                      Carat weight
                      <input
                        name="carat"
                        type="number"
                        min="0"
                        step="any"
                        required
                        value={values.carat}
                        onChange={(event) =>
                          change("carat", event.target.value)
                        }
                      />
                      <span className="field-hint">Weight in carats</span>
                    </label>
                    {["cut", "color", "clarity"].map((name) => (
                      <label className="field" key={name}>
                        {name[0].toUpperCase() + name.slice(1)}
                        <select
                          name={name}
                          value={values[name]}
                          onChange={(event) => change(name, event.target.value)}
                        >
                          {(info?.categories[name] || [values[name]]).map(
                            (value) => (
                              <option key={value}>{value}</option>
                            ),
                          )}
                        </select>
                      </label>
                    ))}
                  </div>
                  <div className="form-divider">
                    <span>PROPORTIONS & DIMENSIONS</span>
                    <span>Millimetres / percent</span>
                  </div>
                  <div className="form-grid">
                    {numericFields
                      .filter((name) => name !== "carat")
                      .map((name) => (
                        <label className="field" key={name}>
                          {fieldLabels[name]}
                          <input
                            name={name}
                            type="number"
                            min="0"
                            max={
                              name === "depth" || name === "table"
                                ? 100
                                : name === "y" || name === "z"
                                  ? 20
                                  : undefined
                            }
                            step="any"
                            required
                            value={values[name]}
                            onChange={(event) =>
                              change(name, event.target.value)
                            }
                          />
                          <span className="field-hint">
                            Training:{" "}
                            {info?.training_numeric_ranges[name]?.min ?? "—"}–
                            {info?.training_numeric_ranges[name]?.max ?? "—"}
                          </span>
                        </label>
                      ))}
                  </div>
                  <button type="submit" className="primary-button">
                    {busy
                      ? "Estimating with both models…"
                      : "Estimate diamond price"}
                    <span aria-hidden="true">↗</span>
                  </button>
                </fieldset>
                <p className="small muted">
                  Inputs are sent unchanged to each model’s saved preprocessing
                  pipeline.
                </p>
              </form>
              <div
                id="comparison"
                className="comparison-panel"
                aria-live="polite"
                aria-busy={busy}
              >
                <div className="panel-title">
                  <h3>Two independent perspectives</h3>
                  <span className="small muted">USD</span>
                </div>
                <article className="estimate primary-estimate">
                  <span className="estimate-label">
                    PRIMARY ESTIMATOR{" "}
                    <span className="pill">Frozen selection</span>
                  </span>
                  <h4>Histogram Gradient Boosting</h4>
                  <strong className="price">
                    {results.ml
                      ? usd(results.ml.predicted_price_usd)
                      : busy
                        ? "Estimating…"
                        : "Ready to estimate"}
                  </strong>
                  <p>Traditional machine learning · persisted full pipeline</p>
                  <span className="model-id">
                    {info?.model_version || "Awaiting verified model"}
                  </span>
                </article>
                <article className="estimate ann-estimate">
                  <span className="estimate-label">
                    DEEP LEARNING COMPARATOR
                  </span>
                  <h4>Deep Neural Network</h4>
                  <strong className="price">
                    {results.ann
                      ? usd(results.ann.predicted_price_usd)
                      : busy
                        ? "Estimating…"
                        : "Awaiting your specifications"}
                  </strong>
                  <p>128 → 64 → 32 → 1 · retained seed 42</p>
                  <span className="model-id">
                    {annInfo?.model_version || "Awaiting verified model"}
                  </span>
                </article>
                {results.ml && results.ann && (
                  <div className="difference">
                    <span>Absolute model difference</span>
                    <strong>
                      {usd(
                        Math.abs(
                          results.ml.predicted_price_usd -
                            results.ann.predicted_price_usd,
                        ),
                      )}
                    </strong>
                    <p>
                      This is model disagreement, not prediction uncertainty.
                      Estimates are not averaged.
                    </p>
                  </div>
                )}
                {errors.map((error) => (
                  <p role="alert" className="error-box" key={error}>
                    {error}
                  </p>
                ))}
                {warnings.length > 0 && (
                  <div className="warning-box">
                    <strong>Review these specifications</strong>
                    <ul>
                      {warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <p className="disclaimer">
                  {results.ml?.disclaimer ||
                    results.ann?.disclaimer ||
                    "Dataset-based USD estimates, not certified appraisals or guaranteed current market values. No individual confidence interval is provided."}
                </p>
              </div>
            </div>
          </section>
          <section id="performance">
            <div className="section-heading">
              <div>
                <p className="eyebrow">MEASURED, NOT ASSUMED</p>
                <h2>Model performance</h2>
              </div>
              <span className="muted small">
                Validation n=8,097 · Test n=8,089
              </span>
            </div>
            <div className="analytics-grid">
              <div className="panel">
                <h3>MAE across evaluation partitions</h3>
                <p className="small muted">
                  USD · lower is better · saved results, no new evaluation
                </p>
                <div className="legend">
                  <span>
                    <i className="legend-validation" />
                    Validation
                  </span>
                  <span>
                    <i className="legend-test" />
                    Final test
                  </span>
                </div>
                {info?.comparison_models.map((metric) => (
                  <div className="chart-group" key={metric.model}>
                    <strong>{modelName(metric.model)}</strong>
                    <div className="chart-row">
                      <div
                        className="bar validation-bar"
                        style={{
                          width: `${(metric.validation_mae_usd / 350) * 100}%`,
                        }}
                      />
                      <span>{usd(metric.validation_mae_usd)}</span>
                    </div>
                    <div className="chart-row">
                      <div
                        className="bar test-bar"
                        style={{ width: `${(metric.mae_usd / 350) * 100}%` }}
                      />
                      <span>{usd(metric.mae_usd)}</span>
                    </div>
                  </div>
                )) || <p>Waiting for verified metrics…</p>}
              </div>
              <div className="panel">
                <h3>Final test comparison</h3>
                <div className="table-scroll">
                  <table>
                    <caption>Frozen candidates · final test results</caption>
                    <thead>
                      <tr>
                        <th scope="col">Model</th>
                        <th scope="col">RMSE (USD)</th>
                        <th scope="col">R²</th>
                        <th scope="col">MAPE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {info?.comparison_models.map((metric) => (
                        <tr key={metric.model}>
                          <th scope="row">{modelName(metric.model)}</th>
                          <td>{usd(metric.rmse_usd)}</td>
                          <td>{metric.r2.toFixed(5)}</td>
                          <td>{metric.mape_percent.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <details className="validation-details">
                  <summary>View validation RMSE and R?</summary>
                  <table>
                    <caption>Validation partition ? n=8,097</caption>
                    <thead>
                      <tr>
                        <th scope="col">Model</th>
                        <th scope="col">RMSE (USD)</th>
                        <th scope="col">R?</th>
                      </tr>
                    </thead>
                    <tbody>
                      {info?.comparison_models.map((metric) => (
                        <tr key={metric.model}>
                          <th scope="row">{modelName(metric.model)}</th>
                          <td>{usd(metric.validation_rmse_usd)}</td>
                          <td>{metric.validation_r2.toFixed(5)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
                <div className="note">
                  <strong>Selection stays frozen.</strong>
                  <p>
                    Boosting was selected using validation MAE before final
                    testing. Small score differences do not establish
                    statistical superiority. MAPE emphasizes lower-priced
                    diamonds.
                  </p>
                </div>
              </div>
            </div>
          </section>
          <section id="insights">
            <div className="section-heading">
              <div>
                <p className="eyebrow">BEHIND THE ESTIMATE</p>
                <h2>What the model relies on</h2>
              </div>
              <span className="muted small">Primary boosting model</span>
            </div>
            <div className="analytics-grid">
              <div className="panel">
                <h3>Permutation feature importance</h3>
                <p className="small muted">
                  {insight?.reference ||
                    "Waiting for verified explanation report…"}
                </p>
                {insight?.importance.map((item) => (
                  <div className="importance-row" key={item.feature}>
                    <span>{item.feature}</span>
                    <div className="importance-track">
                      <div
                        style={{
                          width: `${Math.max(0, (item.mean_mae_increase_usd / insight.importance[0].mean_mae_increase_usd) * 100)}%`,
                        }}
                      />
                    </div>
                    <span>{usd(item.mean_mae_increase_usd)}</span>
                  </div>
                ))}
              </div>
              <div className="panel insight-copy">
                <span className="insight-symbol" aria-hidden="true">
                  ◇
                </span>
                <h3>
                  Importance is a model perspective.
                  <br />
                  Not a causal claim.
                </h3>
                <p>
                  {insight?.limitations ||
                    "Explanations are based on training and validation references, never test-driven feature selection."}
                </p>
                <p>
                  Each original characteristic is permuted through the full
                  pipeline, so all one-hot levels move together.
                  Partial-dependence investigations of y, carat and z are
                  documented in the assessment artifacts; they describe average
                  model responses, not guaranteed individual price changes.
                </p>
              </div>
            </div>
          </section>
          <section id="methodology" className="methodology">
            <p className="eyebrow">ABOUT CARATMIND</p>
            <h2>Built to be understood.</h2>
            <p>
              An AI Engineering assessment connecting data quality, reproducible
              experimentation and transparent inference.
            </p>
            <div className="method-grid">
              <article>
                <span>01 / THE DATA</span>
                <h3>Diamond characteristics</h3>
                <p>
                  53,940 original observations. 23 invalid dimension records
                  removed. 53,917 retained; 48 investigated depth mismatches
                  retained unchanged.
                </p>
              </article>
              <article>
                <span>02 / THE METHOD</span>
                <h3>Separate by design</h3>
                <p>
                  37,731 training, 8,097 validation and 8,089 test rows. Exact
                  input groups stay together. Scaling and encoding were fitted
                  on training only.
                </p>
              </article>
              <article>
                <span>03 / THE LIMITS</span>
                <h3>Estimates with context</h3>
                <p>
                  Historical dataset prices may not reflect current markets.
                  High-price errors are larger; rare grades and unfamiliar
                  combinations need caution. No appraisal guarantee.
                </p>
              </article>
            </div>
            <p className="small methodology-footer">
              ML: tuned HistGradientBoosting · DL: selected seed-42 ANN · Final
              test consumed once · No model training during requests
            </p>
          </section>
          <footer>
            <a className="footer-brand" href="#overview">
              ◇ CaratMind
            </a>
            <span>Evidence-led diamond intelligence.</span>
            <span>Assessment project · local development</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
