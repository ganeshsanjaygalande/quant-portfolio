#!/usr/bin/env python3
"""
etf_gate.py — Component 4: the CPCV validation gate for the NIFTYBEES trend overlay.
================================================================================
Renders the HONEST verdict, per the agreed design:
  * DSR on RAW strategy returns, deflated by the pre-registered K  -> "is the Sharpe real?"
  * PAIRED distribution across CPCV out-of-sample paths of (Sharpe, MaxDD, Calmar) for
    Strategy vs Buy-and-Hold, and Delta MaxDD = MaxDD_BH - MaxDD_strat  -> "does the
    overlay actually cut drawdown, robustly to sequence-of-returns luck?"
Verdict on the mandate = the 5th percentile of Delta MaxDD across paths.
"""
from __future__ import annotations
import os, sys, itertools, inspect
from math import comb
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etf_engine import build_positions, backtest
from research_framework import CombinatorialPurgedCV, deflated_sharpe

ANN = np.sqrt(252)

# ── pre-registered grid (declare K BEFORE running) ───────────────────────────
GRID = [dict(fast=f, slow=s, target_vol=tv)
        for f in (16, 32) for s in (96, 192) for tv in (0.008, 0.012)]   # K = 8
K_BUDGET = len(GRID)
BT_KWARGS = dict(max_cost_ratio=0.005)   # requires the Task-A engine; {} for the base engine

# ── metrics ──────────────────────────────────────────────────────────────────
def _sharpe(r):
    r = np.asarray(r); s = r.std()
    return float(r.mean() / s * ANN) if s > 0 else 0.0
def _maxdd(r):
    eq = np.cumprod(1 + np.asarray(r)); peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / peak)) if len(eq) else 0.0
def _calmar(r):
    r = np.asarray(r); n = len(r)
    if n < 2: return 0.0
    cagr = (1 + r).prod() ** (252 / n) - 1; dd = _maxdd(r)
    return float(cagr / dd) if dd > 1e-9 else np.nan

def run_config(price: pd.Series, cfg: dict, bt_kwargs: dict) -> pd.Series:
    pos = build_positions(price, fast=cfg["fast"], slow=cfg["slow"], target_vol=cfg["target_vol"])
    ok = set(inspect.signature(backtest).parameters)     # tolerate base OR Task-A engine
    kw = {k: v for k, v in bt_kwargs.items() if k in ok}
    bt = backtest(price, pos["exposure"], **kw)
    return bt["strat_ret"]

# ── paired CPCV: per OOS path, Strategy vs Buy-and-Hold ──────────────────────
def cpcv_paired(cfg_ret: pd.DataFrame, bh: pd.Series, N=10, k=2, embargo=0.01):
    R = cfg_ret.values; T = len(cfg_ret)
    groups = np.array_split(np.arange(T), N)
    cv = CombinatorialPurgedCV(n_groups=N, k_test=k, embargo_frac=embargo)
    pg_s = {g: [] for g in range(N)}; pg_b = {g: [] for g in range(N)}
    for tr, te, combo in cv.split(T):
        if len(tr) < 50: continue
        is_sr = R[tr].mean(0) / (R[tr].std(0) + 1e-12)
        best = int(np.argmax(is_sr))
        for g in combo:
            pg_s[g].append(R[groups[g], best]); pg_b[g].append(bh.values[groups[g]])
    n_paths = comb(N, k) * k // N
    rows = []
    for p in range(n_paths):
        if any(len(pg_s[g]) <= p for g in range(N)): continue
        s = np.concatenate([pg_s[g][p] for g in range(N)])
        b = np.concatenate([pg_b[g][p] for g in range(N)])
        rows.append(dict(sharpe_s=_sharpe(s), sharpe_b=_sharpe(b),
                         maxdd_s=_maxdd(s), maxdd_b=_maxdd(b),
                         calmar_s=_calmar(s), calmar_b=_calmar(b),
                         d_maxdd=_maxdd(b) - _maxdd(s)))
    return pd.DataFrame(rows)

def run_gate(price: pd.Series, bt_kwargs: dict = BT_KWARGS):
    price = price.dropna()
    cfg_ret = pd.DataFrame({f"c{i}": run_config(price, c, bt_kwargs) for i, c in enumerate(GRID)}).dropna()
    bh = price.pct_change(fill_method=None).reindex(cfg_ret.index).fillna(0.0)
    paths = cpcv_paired(cfg_ret, bh)
    # DSR on the most-selected config's raw returns, deflated by K
    is_sr = cfg_ret.mean() / cfg_ret.std()
    best = is_sr.idxmax()
    dsr, sr, sr0 = deflated_sharpe(cfg_ret[best].values, K_BUDGET)
    return cfg_ret, bh, paths, dict(best=best, dsr=dsr, sr=sr, sr0=sr0)

if __name__ == "__main__":
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    px = pd.read_csv(os.path.join(_REPO,"data","clean","panel_close.csv"),
                     index_col=0, parse_dates=True)["NIFTYBEES"].dropna()
    cfg_ret, bh, paths, d = run_gate(px)
    p5 = paths["d_maxdd"].quantile(0.05); pmed = paths["d_maxdd"].median()
    win = (paths["d_maxdd"] > 0).mean() * 100
    print("="*70); print("COMPONENT 4 — CPCV GATE  (NIFTYBEES long-only EWMA overlay)")
    print("="*70)
    print(f"DSR (raw returns, deflated by K={K_BUDGET}) = {d['dsr']:.3f}   [Sharpe gate >0.95]")
    print(f"CPCV paths: {len(paths)}")
    print(f"Sharpe  strat vs B&H (median): {paths['sharpe_s'].median():.2f} vs {paths['sharpe_b'].median():.2f}")
    print(f"MaxDD   strat vs B&H (median): {paths['maxdd_s'].median()*100:.1f}% vs {paths['maxdd_b'].median()*100:.1f}%")
    print(f"Calmar  strat vs B&H (median): {paths['calmar_s'].median():.2f} vs {paths['calmar_b'].median():.2f}")
    print(f"Delta MaxDD  median={pmed*100:+.1f}pp   5th-pctile={p5*100:+.1f}pp   paths cutting DD={win:.0f}%")
    dd_ok = p5 > 0
    print(f"\nVERDICT — mandate (DD cut robust at 5th pctile): {'PASS' if dd_ok else 'FAIL'}")
    print(f"         — Sharpe real (DSR>0.95): {'PASS' if d['dsr']>0.95 else 'FAIL / not proven'}")
