# Quantitative Strategy Research & Validation

A two-project portfolio in systematic trading research. The theme is **methodological
rigor over marketing**: strategies are put through leakage-free, out-of-sample statistical
gates and are *rejected* when the evidence says so — and certified only when they clear
pre-registered thresholds.

The through-line across both projects is a reusable validation stack:
- **Causal, look-ahead-free backtesting** (verified by an explicit anti-look-ahead test).
- **Combinatorial Purged Cross-Validation** and the **Deflated Sharpe Ratio / PBO** (Lopez de Prado).
- **Ledoit-Wolf covariance shrinkage**, volatility targeting, realistic transaction-cost models.
- **Data integrity**: deterministic SHA-256 dataset hashing, corporate-action adjustment, and a
  CI gate that blocks merges on data/QC failure or trial-budget breach.

## Projects

### 1 — [Systematic Strategy Research](./01-systematic-strategy-research)
A research program (originally aimed at a prop-firm funding challenge) that tested intraday
statistical arbitrage, single-leg mean-reversion, and vol-targeted time-series momentum on a
macro futures basket. **Headline: honest negative results.** Every candidate was rejected by
the statistical gates — thin edge vs. friction, overfit to a bull market, unproven under purged
CV. The durable output was the validation methodology, carried into Project 2.

### 2 — [NIFTY ETF Trend Overlay](./02-nifty-etf-trend-overlay)
A long-only EWMA trend overlay on an Indian index ETF (NIFTYBEES), built end-to-end from a free
NSE-bhavcopy data pipeline through a CPCV/DSR gate. **Certified result:** roughly *halves*
maximum drawdown vs buy-and-hold (16.9% vs 36.3%), robust across 100% of CPCV out-of-sample
paths — validated as a drawdown-reduction overlay, explicitly *not* as a raw-return alpha engine
(Deflated Sharpe 0.94, just under the 0.95 bar; the honest call is reported as-is, not tuned to cross it).

## Skills demonstrated
Python (numpy/pandas); time-series & backtest engineering without look-ahead; financial data
engineering (ingestion, cleaning, corporate actions, reproducible hashing); CI/CD gating; and
applied statistical validation (purged cross-validation, Deflated Sharpe, PBO, Ledoit-Wolf shrinkage).

## A note on integrity
Developed with AI-assisted pair-programming; every design decision, statistical method, and
result was reviewed and is understood by the author. The defining feature of this work is
intellectual honesty — negative results are reported, not hidden, and no result is tuned to
pass a gate after the fact.

## License
MIT — see [LICENSE](./LICENSE). Educational research; not investment advice.
