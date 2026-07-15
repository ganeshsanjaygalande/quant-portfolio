# MASTER_PLAN.md — Research Charter (LOCKED)
*Owner: Operator · Referee: Claude (CTO) · Strategist: Gemini · v1.0*

## Objective
Build and honestly validate a diversified, **positive-skew** systematic engine to
compound own capital. Success = an edge that clears the pre-registered gates on
leakage-free validation. Nothing deploys that hasn't cleared them.

## Strategy family (LOCKED — changing this is a new charter, not a config)
- **Trend + breakout only.** Buy strength / sell weakness. No mean-reversion.
  Rationale: positive skew (Fung-Hsieh: trend = synthetic long straddle) is the
  correct distribution shape for capital preservation and drawdown control.
- **Signal ensemble** (fixed): TSMOM sign {21,63,126,252}d + EWMA crossover
  {8/24,16/48,32/96} + Donchian breakout {63,126}. Averaged within/across families.
- **Volatility targeting**: size to a fixed portfolio daily-vol budget using a
  Ledoit-Wolf shrunk covariance (`research_framework.ledoit_wolf_cc`).
- **Jump-aware sizing**: per-instrument exposure caps + gap stops for fat-tailed
  markets (energy, crypto); portfolio de-gross when shrunk correlation rises.

## Universe (LOCKED list in PRE_REGISTRATION.md)
~50 liquid futures across 6 sectors (equity indices, rates, FX, energy, metals,
grains/softs/meats). Diversification comes from sector breadth.

## Data contract
- Roll-adjusted continuous futures (Norgate/CSI). Returns computed from
  ratio/roll-adjusted closes ONLY (never Panama back-adjusted — pipeline guards).
- Every dataset is fingerprinted by `data_pipeline.py` -> `dataset_hash`.
- No backtest is valid without a `dataset_hash` recorded in RESULTS_LEDGER.md.

## Gates (absolute)
- **DSR > 0.95** (deflated by pre-registered K).
- **PBO < 5%** (combinatorial purged CV).
- FTMO-style risk sanity retained for own-capital: worst-day and maxDD within
  budget on the training window BEFORE the locked-box test.

## Planes of work
- **Engineering plane** (parallel, agent-heavy, CI-verified): data_pipeline, QC,
  wiring, refactors. Jules/Ruflo operate here.
- **Research plane** (serial, human-gated, K-counted): anything touching the gates.
  Agents may stage runs; only the Operator authorizes one, and every run is logged.
