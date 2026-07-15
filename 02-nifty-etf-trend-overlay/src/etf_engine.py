#!/usr/bin/env python3
"""
etf_engine.py — long-only EWMA trend overlay for a single index ETF (NIFTYBEES).
================================================================================
DEFINITIVE build: Components 1-3 + the Task-C audit fixes integrated & tested:
  - verified equity-ETF COSTS (F1: no buy STT, 0.001% sell STT; F5: DP Rs13 pre-GST)
  - F8 no-borrow guard (buys shrink so cash never goes negative)
  - F9 price-validity check (finite, positive prices required)
  - F10 no-trade band: cost-ratio band + exposure deadband (OR-to-skip), no-op at 0.
Mandate: cut MaxDD/Calmar vs buy-and-hold by sitting in cash through downtrends.
NO alpha stats here — that waits for the CPCV/DSR gate (etf_gate.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Component 1: EWMA crossover signal ───────────────────────────────────────
def ewma_signal(price: pd.Series, fast: int = 32, slow: int = 96,
                vol_win: int = 63) -> pd.Series:
    """Trend signal in [-1, 1]. >0 uptrend, <0 downtrend. Causal, scale-free, tanh-capped."""
    ef = price.ewm(span=fast, adjust=False).mean()
    es = price.ewm(span=slow, adjust=False).mean()
    raw = ef - es
    norm = raw / price.rolling(vol_win).std()
    return np.tanh(norm).rename("signal")


# ── Component 2: long-only, vol-targeted exposure ────────────────────────────
def target_exposure(price: pd.Series, signal: pd.Series,
                    target_vol: float = 0.01, max_leverage: float = 1.0,
                    vol_win: int = 20) -> pd.Series:
    """Signal -> fraction of equity in the ETF, in [0, max_leverage]. Long-only, vol-targeted."""
    ret = price.pct_change(fill_method=None)
    rv = ret.ewm(span=vol_win).std()
    long_sig = signal.clip(lower=0.0)
    expo = long_sig * (target_vol / rv)
    expo = expo.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return expo.clip(0.0, max_leverage).rename("exposure")


def build_positions(price: pd.Series, fast: int = 32, slow: int = 96,
                    target_vol: float = 0.01, max_leverage: float = 1.0,
                    sig_vol_win: int = 63, pos_vol_win: int = 20) -> pd.DataFrame:
    sig = ewma_signal(price, fast, slow, sig_vol_win)
    expo = target_exposure(price, sig, target_vol, max_leverage, pos_vol_win)
    return pd.DataFrame({"price": price, "signal": sig, "exposure": expo})


# ── Component 3: causal share-level backtest + India cost stack ──────────────
# VERIFIED (Task-C audit) equity-oriented-ETF rates: NIFTYBEES on NSE, delivery, CDSL.
COSTS = dict(
    stt_buy=0.0,           # F1: NIL STT on ETF purchases
    stt_sell=0.00001,      # F1: 0.001% STT, sell side only, delivery
    exch_txn=0.0000307,    # F2: NSE cash 0.00297% + IPFT = 0.00307% per side
    sebi=0.000001,         # F3: SEBI turnover fee Rs10/crore, both sides
    stamp_buy=0.00015,     # F4: 0.015% stamp duty, buy side only
    gst=0.18,              # F6: 18% GST on (brokerage + exch_txn + sebi) and on DP fee
    brokerage_per_order=0.0,   # F7: discount-broker equity delivery = Rs0
    dp_per_sell=13.0,      # F5: pre-GST DP fee per sell-day; x1.18 = Rs15.34 all-in
)

def _trade_cost(trade_val: float, is_sell: bool, c: dict) -> float:
    brokerage = c["brokerage_per_order"]
    txn = trade_val * c["exch_txn"]
    sebi = trade_val * c["sebi"]
    stt = trade_val * (c["stt_sell"] if is_sell else c["stt_buy"])
    stamp = 0.0 if is_sell else trade_val * c["stamp_buy"]
    gst = c["gst"] * (brokerage + txn + sebi)
    dp = c["dp_per_sell"] * (1 + c["gst"]) if is_sell else 0.0
    return brokerage + txn + sebi + stt + stamp + gst + dp


def backtest(price: pd.Series, exposure: pd.Series, capital: float = 100000.0,
             costs: dict = COSTS, max_cost_ratio: float = 0.0,
             deadband: float = 0.0) -> pd.DataFrame:
    """Causal share-level simulation (position set at close t earns t+1).
       F9: validate prices. F8: never let cash go negative. F10: optional no-trade band —
       skip a rebalance if EITHER the estimated cost exceeds `max_cost_ratio` x trade value
       OR the exposure change since the last executed trade is below `deadband`.
       Both bands default to 0.0 (no-op). Returns equity/cost/pos_frac/strat_ret; no alpha."""
    px = price.values.astype(float)
    if not (np.isfinite(px).all() and (px > 0).all()):      # F9
        raise ValueError("backtest requires finite, positive prices (dropna first)")
    ex = exposure.reindex(price.index).fillna(0.0).values
    n = len(px)
    shares = 0
    cash = float(capital)
    last_exec_expo = 0.0
    equity = np.empty(n); daily_cost = np.zeros(n); pos_val = np.empty(n)
    for t in range(n):
        eq_now = cash + shares * px[t]
        target_sh = int(np.floor(ex[t] * eq_now / px[t]))
        delta = target_sh - shares
        # F8: shrink a buy until share value + cost fits in cash (no borrowing)
        if delta > 0:
            while delta > 0:
                cst = _trade_cost(delta * px[t], is_sell=False, c=costs)
                if cash - delta * px[t] - cst >= 0:
                    break
                delta -= 1
        # F10: no-trade band (OR-to-skip)
        if delta != 0:
            trade_val = abs(delta) * px[t]
            est_cost = _trade_cost(trade_val, is_sell=(delta < 0), c=costs)
            drift = abs(ex[t] - last_exec_expo)
            if (max_cost_ratio > 0 and est_cost > max_cost_ratio * trade_val) or \
               (deadband > 0 and drift < deadband):
                delta = 0
        # execute
        if delta != 0:
            trade_val = abs(delta) * px[t]
            cst = _trade_cost(trade_val, is_sell=(delta < 0), c=costs)
            cash -= delta * px[t] + cst
            daily_cost[t] = cst
            shares += delta
            last_exec_expo = ex[t]
        equity[t] = cash + shares * px[t]
        pos_val[t] = shares * px[t]
    out = pd.DataFrame(index=price.index)
    out["equity"] = equity
    out["cost"] = daily_cost
    out["pos_frac"] = pos_val / equity
    out["strat_ret"] = out["equity"].pct_change(fill_method=None).fillna(0.0)
    return out


if __name__ == "__main__":
    import os
    px = pd.read_csv(os.path.join("data","clean","panel_close.csv"),
                     index_col=0, parse_dates=True)["NIFTYBEES"].dropna()
    pos = build_positions(px)
    bt = backtest(px, pos["exposure"], capital=100000.0, max_cost_ratio=0.005)
    yrs = (px.index[-1]-px.index[0]).days/365.25
    trades = int((bt["cost"] > 0).sum()); tot_cost = bt["cost"].sum()
    print(f"MECHANISM/COST DIAGNOSTICS (verified ETF costs; band max_cost_ratio=0.005)")
    print(f"  span            : {px.index[0].date()} -> {px.index[-1].date()} ({yrs:.1f}y)")
    print(f"  rebalances      : {trades}")
    print(f"  total cost      : Rs {tot_cost:,.0f} on Rs 1,00,000 start")
    print(f"  cost drag       : {1e4*tot_cost/100000/yrs:.0f} bps/yr")
    print(f"  min cash (>=0?) : Rs {(bt['equity']-bt['pos_frac']*bt['equity']).min():,.2f}")
