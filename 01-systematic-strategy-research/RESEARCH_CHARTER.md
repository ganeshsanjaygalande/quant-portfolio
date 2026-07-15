# Large-Universe Trend Study — Research Charter

**Objective:** Build and honestly validate a diversified, positive-skew, systematic
trend/momentum engine to compound *own capital* — not to sprint a prop barrier.
Success = an out-of-sample edge that survives Combinatorial Purged CV and clears a
**pre-registered** Deflated Sharpe gate. Deployed only after it clears.

---

## 1. The Universe (~50 liquid futures, 6 sectors)

Diversification comes from *sector breadth* — ags barely correlate with rates, energy
with equities, etc. Target ~45–55 markets. Micro contracts (MES/MNQ/MYM/M2K, micro
FX, micro gold) let a small account hold min-size positions.

**Equity indices (9):** ES (S&P500), NQ (Nasdaq), YM (Dow), RTY (Russell), FESX (Euro Stoxx),
FDAX (DAX), Z (FTSE100), NKD (Nikkei), HSI (Hang Seng).

**Rates / fixed income (9):** ZT (2Y), ZF (5Y), ZN (10Y), ZB (30Y), UB (Ultra), SR3 (SOFR),
FGBL (Bund), FGBM (Bobl), FGBS (Schatz). *(Add Long Gilt / JGB if data allows.)*

**FX (8):** 6E (EUR), 6J (JPY), 6B (GBP), 6A (AUD), 6C (CAD), 6S (CHF), 6N (NZD), 6M (MXN).

**Energy (5):** CL (WTI), BRN (Brent), NG (NatGas), RB (RBOB), HO (Heating Oil).

**Metals (5):** GC (Gold), SI (Silver), HG (Copper), PL (Platinum), PA (Palladium).

**Grains / softs / meats (12):** ZC (Corn), ZW (Wheat), ZS (Soybean), ZL (Soy Oil),
ZM (Soy Meal), SB (Sugar), KC (Coffee), CC (Cocoa), CT (Cotton), LE (Live Cattle),
HE (Lean Hogs), GF (Feeder Cattle).

**Optional 7th sleeve:** BTC/ETH futures — high vol, cap the weight hard.

---

## 2. Data sourcing (Norgate or CSI)

- **Norgate Data** (Futures package) or **CSI (UA)** — both give *back-adjusted continuous
  contracts* with roll handling and full multi-decade history. Norgate integrates cleanly
  with Python (`norgatedata`), supports Panama (difference) and ratio adjustment, and gives
  roll dates.
- **CRITICAL gotcha:** never compute % returns off a back-adjusted price series — Panama
  adjustment shifts levels and can even go negative, so `pct_change()` is meaningless. Use
  one of: (a) ratio-adjusted series for returns, or (b) unadjusted front-contract prices
  with explicit roll-gap handling. Get *both* adjusted (for signals) and the return stream
  (roll-adjusted) from the vendor.
- Store: daily OHLCV per continuous series + roll dates + per-contract multiplier/tick value
  (needed for real position sizing and cost).

---

## 3. Execution & capital reality (IBKR)

- **IBKR** replaces MetaApi/MT5: real futures, proper margin, `ib_insync`/TWS API, far better
  fills and cost than CFDs. This alone removes a large chunk of the friction that killed the
  CFD tests.
- **Capital honesty:** a true ~50-market book at sane risk needs more than $50k to hold
  ≥1 contract everywhere. Two fixes: (a) trade *micros* where they exist; (b) run a *rotating
  sub-basket* — hold the N markets with the strongest signals, not all 50 at once. Even so,
  budget realistically ($50k is thin for full breadth; the study is valid at any size but
  live sizing scales with capital).

---

## 4. The Mathematics (code in `research_framework.py`)

**Signal ensembling — `ensemble_forecast()`.** Three trend *families*, each a single idea:
(1) time-series momentum = sign of trailing return over {21,63,126,252}d; (2) EWMA crossover
{8/24,16/48,32/96} normalized by its own vol (Baz/AQR style); (3) Donchian breakout {63,126}.
Average within family, then across families -> one forecast in [-1,1] per market. Ensembling
is variance reduction, **not** new free parameters — the lookback sets are *fixed by charter*.

**Covariance — `ledoit_wolf_cc()`.** Sample covariance of ~50 markets on limited history is
ill-conditioned and unstable. Ledoit-Wolf shrinks it toward a constant-correlation target with
the analytically optimal intensity delta. Use the shrunk Sigma for (a) portfolio-vol targeting
`w'Sigma w = V_target^2` and (b) optional risk-parity weights. Shrinkage value *rises* with N —
negligible at 6 assets (delta≈0.05), material at 50.

**Sizing.** Per-market: forecast x (sigma_target / sigma_i) (inverse-vol). Portfolio: scale so
ex-ante vol from Sigma_shrunk = budget. Per-instrument jump caps + portfolio de-gross when the
shrunk correlation rises (crisis clustering). Target daily vol conservative (0.3–0.5%).

**Validation — `CombinatorialPurgedCV`.** Replace the single split with N-group, k-test
combinatorial folds (purge + embargo each). N=6,k=2 -> 15 splits / 5 backtest *paths*; scale
up (e.g. N=10,k=2 -> 45 splits / 9 paths). This yields a **distribution** of OOS Sharpe, from
which you compute the **Probability of Backtest Overfitting (PBO)** = fraction of paths where
the in-sample-best config underperforms OOS-median. Gate: **PBO < 5%**.

**Deflated Sharpe — `deflated_sharpe()`.** Feed the CPCV path variance and the pre-registered
trial count K. Gate: **DSR > 0.95**.

---

## 5. The Trials Budget — PRE-REGISTER BEFORE RUNNING A SINGLE BACKTEST

The failure mode is never bad math; it is testing until something passes. Lock this first:

| Item | Locked value |
|---|---|
| Universe | the ~50 markets above (no additions later) |
| Signal families | 3 (TSMOM, EWMA-cross, breakout) |
| Lookback sets | as listed above (fixed) |
| Vol targets tested | {0.3%, 0.4%, 0.5%} |
| Portfolio methods | {inverse-vol, LW risk-parity} |
| **Total trials K** | 3 families-on/off combos not counted; **config grid = lookback-sets(1 locked) x vol(3) x method(2) = 6** core configs. If you also test 3 embargo settings -> K=18. **Declare K and never exceed it.** |
| Hold-out | most recent **20%** of history = locked box, touched **once**, at the very end |
| DSR gate | > 0.95 (deflated by K) |
| PBO gate | < 5% |
| Stop rule | if it fails both gates, it FAILS. Adding "one more idea" retroactively inflates K and voids the test. |

Write K on paper before you start. Every reported Sharpe is deflated by that K, forever.

---

## 6. Phased plan with go/no-go gates

1. **Data & plumbing** — Norgate pull, roll-adjusted return streams, cost/multiplier table,
   QC every series (the referee QC we already do). *Gate: clean panel, no survivorship gaps.*
2. **Signal + portfolio build** — wire `ensemble_forecast` + `ledoit_wolf_cc` + sizing into
   the engine. *Gate: risk targeting holds (worst day < daily budget x ~4 in-sample).*
3. **CPCV validation** — run the pre-registered K configs through CombinatorialPurgedCV.
   *Gate: DSR > 0.95 AND PBO < 5% on the training 80%.*
4. **Locked-box test** — run the single surviving config ONCE on the held-out 20%.
   *Gate: OOS Sharpe within the CPCV distribution; no degradation.*
5. **Paper trade** 3–6 months on IBKR paper. *Gate: live slippage ~ modeled.*
6. **Deploy small**, scale only with realized (not backtested) performance.

---

## 7. Honest expectations

- A diversified trend engine's *realistic* Sharpe is ~0.7–1.1 gross, less after cost. That
  compounds own capital respectably over years; it does **not** produce prop-sprint returns.
- With ~50 markets and limited *paths*, even a real edge may not clear DSR 0.95 — that's the
  gate's job, and a fail means "not proven," not "definitely worthless."
- Positive skew means long strings of small losses between wins. Budget psychologically for it.
- Most of the value already banked: the **validation discipline**. Guard it.
