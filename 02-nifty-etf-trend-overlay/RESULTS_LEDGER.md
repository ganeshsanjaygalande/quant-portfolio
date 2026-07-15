# RESULTS_LEDGER.md — Every research-plane run, ever. (APPEND-ONLY)
*A run that is not in this ledger did not happen. K is counted by rows here.*

## How to read
- **trial_id**: T01..T18 (enumerated) or R1/R2 (reserved). One per K-slot; reruns
  of the SAME config on the SAME dataset_hash reuse the id (cache hit, no new K).
- **dataset_hash**: from `data_pipeline.py`. If the current data's hash != the row's
  hash, that row is **STALE** and its verdict is void until re-run.
- **status**: PENDING · PASS · REJECT · STALE.
- Gates: DSR > 0.95 AND PBO < 5%.

| trial_id | date_utc | commit | dataset_hash (12) | config (vol/sizing/signals) | CPCV Sharpe [min..med..max] | DSR | PBO | worst_day | maxDD | status | referee note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| _ex_ | 2026-06-08 | a1b2c3d | 9cd1fe9c4a60 | 0.40% / invvol / TSMOM+EWMA+BO | [0.31..0.72..1.10] | 0.41 | 22% | -2.1% | 9.4% | REJECT | DSR<0.95; PBO>5% — selection noise |

<!-- Append real rows below. Never edit a past row except to flip status to STALE. -->

## Current dataset
- dataset_hash: `bea7e73dc3d5`
- schema_version: 1.0.0
- K spent: 0 / 20

## NIFTYBEES long-only EWMA trend overlay — Component 4 verdict (2026-07)
| trial | dataset_hash | config (CV-selected) | DSR (K=8) | ΔMaxDD med / 5th-pctile | MaxDD strat vs B&H | Sharpe strat vs B&H | Calmar strat vs B&H | status |
|---|---|---|---|---|---|---|---|---|
| ETF-overlay v1 | bea7e73dc3d5 | best-of-8 EWMA (fast/slow/tgtvol), band max_cost_ratio=0.005 | 0.940 | +19.5pp / +18.9pp (100% paths cut DD) | 16.9% vs 36.3% | 0.70 vs 0.75 | 0.34 vs 0.29 | MANDATE PASS (drawdown cut certified, robust OOS); raw-Sharpe edge NOT certified (DSR 0.940<0.95) |

**Honest read:** certified as a *drawdown-reduction overlay* on Indian equity beta — roughly halves MaxDD vs buy-and-hold, robust across all CPCV paths — NOT as an alpha engine (no raw-return edge; DSR just under 0.95). Costs (44 bps/yr, verified ETF stack) included. Single market, single 16.5y sample: modest breadth; do not over-generalize.
