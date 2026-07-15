# NIFTY ETF Trend Overlay — a certified drawdown-reduction engine

A long-only EWMA trend overlay on the Indian index ETF **NIFTYBEES**, built end-to-end from a
free NSE data pipeline through a leakage-free statistical gate, and validated over 16.5 years
(2010-2026).

## Result — certified, out-of-sample (Combinatorial Purged CV)
| Metric | Overlay | Buy & Hold |
|---|---|---|
| Median Max Drawdown | **16.9%** | 36.3% |
| Median Sharpe | 0.70 | 0.75 |
| Median Calmar | **0.34** | 0.29 |
| ΔMaxDD (median / 5th-pctile) | **+19.5pp / +18.9pp** | — |
| CPCV paths that cut drawdown | **100%** | — |
| Deflated Sharpe (deflated by K=8) | 0.94 | — |

**Honest verdict:** certified as a **drawdown-reduction overlay** — it robustly ~halves the
worst crash vs holding the index — but **not** as a raw-return alpha engine. There is no
raw-return edge (Sharpe slightly below buy-and-hold), and the Deflated Sharpe falls just short
of the 0.95 bar. Reported exactly as measured; **not** re-tuned to cross the line.
Full run: [`docs/sample_gate_output.txt`](./docs/sample_gate_output.txt).

## Architecture
```
NSE bhavcopy (free)  ->  data_pipeline (QC + SHA-256 hash)  ->  etf_engine (EWMA signal,
vol-target, integer shares, audited ETF cost stack)  ->  etf_gate (Combinatorial Purged CV +
Deflated Sharpe, paired vs buy-and-hold)  ->  RESULTS_LEDGER
        \___________________ verify_ledger + GitHub Actions CI gate ___________________/
```

## Key modules
- `src/nse_adapter.py` — dual-format (modern UDiFF + legacy) NSE bhavcopy downloader; split
  (corporate-action) adjustment; immutable local cache; weekday-404 anomaly guard.
- `src/data_pipeline.py` — QC + deterministic SHA-256 dataset hashing (any data change -> new
  hash -> prior results auto-invalidated).
- `src/etf_engine.py` — causal backtest: EWMA trend signal, long-only volatility targeting,
  integer shares, **auditor-verified** Indian ETF transaction costs, no-borrow guard, deadband.
- `src/etf_gate.py` — Combinatorial Purged CV + Deflated Sharpe; paired Sharpe/MaxDD/Calmar
  vs buy-and-hold; ΔMaxDD confidence interval across OOS paths.
- `.github/workflows/pre_flight.yml` + `src/verify_ledger.py` — CI gate: blocks a merge on QC
  failure, dataset-hash mismatch, or trial-budget breach.

## Reproduce
1. Supply NIFTYBEES daily bars into `data/raw/` (or run `src/nse_adapter.py` to build them free from NSE).
2. `python src/data_pipeline.py --raw data/raw --out data/clean --freq D`
3. `python src/etf_gate.py`

## Governance
`MASTER_PLAN.md` (research charter) · `PRE_REGISTRATION.md` (trials budget + gates, declared
before running) · `RESULTS_LEDGER.md` (every validated run, dataset-hash stamped).

*Educational research; not investment advice.*
