# Systematic Strategy Research — Stat-Arb & Time-Series Momentum

A research program that rigorously tested several systematic strategies and reported the
**honest negative results** — the outcome most trading portfolios quietly omit.

## What was tested
1. **Intraday statistical arbitrage (index pairs, e.g. NQ/YM).** A both-legs edge-vs-cost
   screen with a real broker-spread friction model across candidate pairs (indices, metals, FX).
   *Finding:* convergence is real but *smaller than two legs of friction* at 5-minute frequency —
   the net edge dies at ~1.9x quoted cost. `src/screen.py`, `results/screen_results.csv`
2. **Single-leg mean-reversion overlay.** Ornstein-Uhlenbeck optimal-entry analysis (Bertram 2010)
   plus the Deflated Sharpe Ratio. *Finding:* apparent profitability was overfit to a bull market —
   DSR ~0.10, and it lost on both long and short legs out-of-sample. `src/v3bt.py`
3. **Vol-targeted time-series momentum, macro futures basket.** Ledoit-Wolf shrinkage,
   combinatorial purged cross-validation, PBO. *Finding:* unproven — the best-selected config's
   Sharpe sat *below* the selection-noise floor. `src/tsmom_engine.py`, `tsmom_macro.py`, `tsmom_v3.py`

## Conclusion
No fundable retail edge at these frequencies/instruments. The value delivered was the reusable
**validation methodology** (purged CV, Deflated Sharpe, PBO, honest friction modeling), which was
then used to certify a real result in [Project 2](../02-nifty-etf-trend-overlay).

## Key files
- `src/research_framework.py` — Ledoit-Wolf shrinkage, signal ensemble, Combinatorial Purged CV, DSR
- `src/screen.py` — both-legs edge-vs-cost statistical-arbitrage screen
- `src/v3bt.py` — single-leg backtest with Bertram OU + Deflated Sharpe
- `src/tsmom_*.py` — vol-targeted TSMOM engine and CPCV/DSR gate
- `src/mt5_extract.py` — MetaTrader5 historical-data extractor
- `RESEARCH_CHARTER.md` — the pre-registered study plan

*Educational research; not investment advice.*
