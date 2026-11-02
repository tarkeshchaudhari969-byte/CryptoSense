# Changelog

## Investor Intelligence Beta
- Added Investor Risk Lab API and dashboard tab.
- Added historical portfolio volatility, drawdown, empirical VaR/CVaR, concentration, correlations and a transparent diversification heuristic.
- Added hypothetical portfolio stress scenarios, fee/slippage friction estimates and risk-budget/stop-distance sizing arithmetic.
- Added `/health`, Dockerfile and Render Blueprint for single-instance deployment.
- Added risk-engine and API integration tests.
- Documented product positioning, limitations, privacy boundaries and a staged monetization plan.

## Investor Intelligence Beta — UX and Reports update
- Rebuilt the frontend with a responsive dark dashboard layout for desktop, tablet and mobile screens.
- Added interactive price-chart pins with per-asset browser persistence, removal and clear-all controls.
- Added eight PDF report periods: 1 day, 1 week, 1 month, 3 months, 6 months, 1 year, 5 years and all available history.
- Added a ReportLab PDF generator with historical performance, risk statistics, a price chart, data-quality checks, provenance and recent OHLCV rows.
- Expanded supported dashboard date windows and made one-observation EDA outputs safe for the 1-day range.
- Added a public-deployment CSV-upload guard, frontend capability configuration and basic response security headers.
- Updated the Render Blueprint to disable shared CSV uploads by default, and documented the remaining steps required before a multi-user SaaS launch.
