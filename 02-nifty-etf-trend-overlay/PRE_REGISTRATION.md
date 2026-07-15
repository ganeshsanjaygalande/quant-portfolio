# PRE_REGISTRATION.md — Trials Budget (APPEND-ONLY; changes are commits)
*Locked before any research-plane run. Every reported Sharpe is deflated by K.*

## K-BUDGET:  K = 20   (18 enumerated + 2 reserved contingency)
The Deflated Sharpe null is computed with **K = 20**. Do not exceed it. Using a
reserved slot requires a logged justification commit. A 21st config VOIDS the
study and forces re-deflation.

### Enumerated grid (18 = 3 x 2 x 3)
| Axis | Values | n |
|---|---|---|
| Portfolio daily-vol target | 0.30% · 0.40% · 0.50% | 3 |
| Sizing method | inverse-vol · Ledoit-Wolf risk-parity | 2 |
| Signal set | TSMOM-only · TSMOM+EWMA · TSMOM+EWMA+Breakout | 3 |

Fixed-by-charter (NOT free parameters, do not sweep): the lookback sets, EWMA
spans, breakout windows, embargo fraction (1%), CPCV geometry (see below).

### Reserved slots (2)
R1, R2 — held for one justified robustness variant each (e.g., alternate vol-est
window). Must be declared in a commit BEFORE running.

## Validation geometry (LOCKED)
- **Combinatorial Purged CV**: N_groups = 10, k_test = 2  -> 45 splits, 9 paths.
- **Embargo**: 1% of T (AFML Ch.7).  **Purge**: label-horizon overlap.
- **Hold-out (locked box)**: most recent **20%** of history. Touched EXACTLY ONCE,
  on the single config that survives the gates on the training 80%.

## Gates (must pass BOTH on training 80%, then confirm on locked box)
- DSR > 0.95  (deflated by K = 20)
- PBO < 5%    (fraction of CPCV paths where IS-best underperforms OOS-median)
- Risk sanity: worst-day loss < daily-vol-budget x 5 ; maxDD < 2x annual-vol.

## Stop rule
Fail both gates -> the study FAILS. No "one more idea." Adding configs after the
fact inflates K retroactively and is scientific misconduct (ASA 2016).

## Sign-off
- [ ] Universe locked (see MASTER_PLAN)
- [ ] K = 20 acknowledged by Operator
- [ ] dataset_hash recorded before first run
