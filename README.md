# CryptoSense AI — Investor Intelligence Beta

CryptoSense AI is a cryptocurrency research and decision-support web application. It combines historical market analytics, machine-learning experiments, anomaly detection, portfolio risk calculations, chart price pins and downloadable PDF research reports.

It is designed as an educational/research beta. It does **not** execute trades, custody assets, connect wallets, provide personalised investment advice, or guarantee returns.

## What's included in this build

### Product experience
- Responsive dark investor dashboard for desktop, tablet and mobile screens.
- BTC, ETH and SOL selector with 1 week, 1 month, 3 months, 6 months, 1 year, 5 years and all available history windows.
- Interactive OHLC / closing-price charts, moving averages, volume, volatility and return distributions.
- **Chart price pins:** click/tap a chart point to pin its actual daily closing price. Pins are saved in the current browser/device, separated by asset, and can be removed or cleared. Pins outside the currently selected chart window remain saved but are hidden until that period is selected again.
- **PDF Research Reports:** one-click exports for 1 day, 1 week, 1 month, 3 months, 6 months, 1 year, 5 years and all available history.
- CSV upload for local/private deployments. The UI reads the deployment capability setting and disables the upload control when uploads are not permitted.

### PDF report contents
Each report includes the selected asset and requested period, actual date coverage, data-source provenance, observation count, latest close, period return, high/low, annualized volatility, maximum drawdown, daily-return statistics, positive-day share, average volume, a vector price-history chart, data-quality checks, recent daily OHLCV observations and methodology/limitations.

Reports are built from the actual data available to the application. The public market-data pipeline currently provides daily candles; therefore the 1 Day report represents the latest available daily observation, not intraday data. `All Time` means all history available from the selected provider, cache or uploaded dataset; it is not a claim of complete asset lifetime history. The current OHLCV pipeline does not supply tokenomics, circulating supply, market capitalization, on-chain data or news sentiment, so those are not fabricated or represented as report content.

### Data Science and analytics
- Data cleaning, validation, provenance and data-quality reporting.
- Exploratory data analysis and descriptive statistics.
- Feature engineering (returns, lags, moving averages, RSI, MACD, volatility, momentum and volume features).
- Chronological machine-learning training/evaluation for classification and regression, with baseline comparisons and permutation importance.
- Z-score, modified-Z, IQR and Isolation Forest anomaly detection.
- K-Means clustering for descriptive market-pattern exploration.
- Multi-asset comparison, normalized performance and return correlation.
- Portfolio volatility, drawdown, empirical VaR/CVaR, concentration, diversification and correlation analysis.
- Hypothetical portfolio stress scenarios, fee/slippage estimates and risk-budget position-sizing arithmetic.

## Technology stack

- Backend: Python, FastAPI, Pandas, NumPy, SciPy and scikit-learn.
- Frontend: HTML, CSS and JavaScript.
- Charts: Plotly.js (loaded from its CDN, so the browser needs internet access to render charts).
- PDF generation: ReportLab.
- Tests: Python `unittest` and FastAPI `TestClient`.
- Deployment: Docker and Render Blueprint configuration.

## Run locally — Windows PowerShell

Open PowerShell in the folder containing `requirements.txt` and `app/`.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use `.env.example` as a reference for environment settings. PowerShell does not load that file automatically, so set the local upload flag in the current session:

```powershell
$env:ENABLE_CSV_UPLOADS="true"
```

Run the app:

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

The application first tries public market-data providers and then clearly-labelled cache or bundled history. If the providers are inaccessible and no local data exists, upload a valid CSV in a private/local deployment. The application does not silently fabricate market prices.

### CSV format

CSV files must be UTF-8 and include `date` and `close`. The optional columns are `open`, `high`, `low` and `volume`. The upload limit is 5 MB and the file must contain at least 30 valid observations. When uploads are enabled, uploaded files are stored by symbol on the server; they are not currently separated by user account.

## Run tests

```powershell
python -m unittest discover tests -v
```

The test fixtures use synthetic price series strictly for testing; synthetic data is never presented as live market prices by the application.

## Deploy a public beta on Render

1. Push this project folder to a Git repository.
2. In Render, create a new Blueprint and point it at that repository.
3. Render reads `render.yaml`, builds the Docker image, mounts persistent storage at `/app/data`, and uses `/health` for the health check.
4. After deployment, check `/health`, then test the dashboard and generate a PDF from the Reports section.

The Render Blueprint intentionally sets `ENABLE_CSV_UPLOADS=false`. That is a safety guard: current CSV uploads are stored globally by asset symbol and are **not isolated by visitor**. Public users must not be able to overwrite one another's dataset. The upload button is disabled in the deployed UI. Keep it disabled until authentication and per-user upload/storage isolation are implemented. For local/private use, set `ENABLE_CSV_UPLOADS=true`.

### Deployment configuration

- `Dockerfile` installs dependencies and starts Uvicorn on the platform `PORT`.
- `render.yaml` provides a Docker-based web service, persistent disk and health check.
- `.dockerignore` excludes local virtual environments, secrets, bytecode, cache and uploaded files from the image build.
- `ENABLE_CSV_UPLOADS` controls the upload API and dashboard control. Local default: `true`; Render public-beta default: `false`.

## Public launch readiness — what is and is not production-ready

This build is deployable as a **single-instance public demo/beta**. It is not yet a secure multi-tenant SaaS product. Before taking payments, collecting user accounts or allowing public uploads, implement and review:

1. Authentication, account recovery, session security and role-based access.
2. Per-user database-backed portfolios, uploads and saved chart pins (pins currently live in the user's browser local storage).
3. Rate limiting and abuse protection on ML training, reports and other compute-heavy endpoints.
4. Database migrations, backup/restore, retention controls and operational monitoring.
5. HTTPS/domain configuration, privacy policy, terms, data-provider attribution and legal/regulatory review appropriate to the countries served.
6. Security testing, dependency scanning, error monitoring and an incident-response process.
7. Automated billing entitlements only after a paid tier and its server-side limits are implemented.

Do not market the model as a price oracle or sell guaranteed-return claims. Report outputs are descriptive historical analytics, not buy/sell signals.

## API routes

- `GET /api/data` — historical market data, quality report and EDA.
- `GET /api/features` — engineered features; `fmt=csv` exports them.
- `GET /api/train` — chronological classification/regression evaluation; `fmt=csv` exports predictions.
- `GET /api/compare` — normalized performance and correlation.
- `GET /api/anomalies`, `GET /api/clusters` — pattern analysis.
- `POST /api/risk-lab` — portfolio risk calculations.
- `GET /api/reports/{symbol}?period=1d|1w|1m|3m|6m|1y|5y|all` — downloadable PDF research report.
- `GET /api/clean.csv` — cleaned market-data CSV.
- `POST /api/upload`, `DELETE /api/upload` — enabled only when `ENABLE_CSV_UPLOADS=true`.
- `GET /api/config` — frontend feature flags.
- `GET /health` — deployment health check.

## Method notes and limitations

- Classification target: `1 if close[t+h] > close[t]`; regression target: `close[t+h]/close[t] - 1`. Features use data available at time `t` only.
- Train/test split is chronological, with a horizon gap; preprocessing is fitted on training data only. Results are compared with simple baselines. Historical scores do not demonstrate a durable trading edge.
- Portfolio risk uses overlapping daily return observations and current user-entered weights held constant through the historical window. VaR/CVaR are empirical estimates, not maximum-loss guarantees. Scenario shocks are hypothetical and not probability-weighted.
- Position sizing is arithmetic, not a recommendation. Stop orders can execute at worse prices; gaps, slippage, liquidity, taxes, funding, exchange rules and market impact can increase losses.
- Data-source availability, historical depth and rate limits can change. The report always labels actual coverage and source so a shorter-than-requested range is visible.
