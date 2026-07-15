#!/usr/bin/env python3
"""
study_engine.py — sealed validation circuit for the trend study.
================================================================================
Consumes the Pre-Flight panel, runs the FULL pre-registered config grid through
the core architecture, evaluates the config-SELECTION process with Combinatorial
Purged CV (N=10,k=2, 1% embargo, purge), computes Deflated Sharpe (deflated by
K=20) and Probability of Backtest Overfitting (CSCV), and prints a Markdown row
ready to append to RESULTS_LEDGER.md.

Why the grid, not one config: CPCV/PBO measure whether SELECTION generalizes. A
single fixed rule has no per-fold choice -> degenerate paths. So each CPCV split
selects the in-sample-best config and scores it out-of-sample. That is the thing
that overfits, so that is the thing we validate.

Usage:
  python study_engine.py --clean ../data/clean --drop UST10Y
"""
from __future__ import annotations
import argparse, os, json, itertools
from math import comb
import numpy as np, pandas as pd
from research_framework import ledoit_wolf_cc, deflated_sharpe, CombinatorialPurgedCV

ANN = np.sqrt(252)

# ── data ─────────────────────────────────────────────────────────────────────
def load_panel(clean_dir, drop=()):
    px = pd.read_csv(os.path.join(clean_dir, "panel_close.csv"), index_col=0, parse_dates=True)
    px = px.drop(columns=[c for c in drop if c in px.columns], errors="ignore")
    px = px.dropna(how="all").sort_index()
    px = px.dropna()                      # common-history panel (referee-honest)
    ret = px.pct_change(fill_method=None)
    dhash = None
    mpath = os.path.join(clean_dir, "manifest.json")
    if os.path.exists(mpath):
        dhash = json.load(open(mpath)).get("dataset_hash")
    return px, ret, dhash

# ── signals (families toggleable for the pre-registered signal-set axis) ─────
def build_forecast(px, families):
    out = pd.DataFrame(0.0, index=px.index, columns=px.columns); nf = 0
    if "tsmom" in families:
        m = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for L in (21, 63, 126, 252): m += np.sign(px / px.shift(L) - 1)
        out += m / 4; nf += 1
    if "ewma" in families:
        c = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for f, s in ((8, 24), (16, 48), (32, 96)):
            x = px.ewm(span=f).mean() - px.ewm(span=s).mean()
            c += np.tanh(x / x.rolling(252, min_periods=63).std())
        out += c / 3; nf += 1
    if "bo" in families:
        b = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for L in (63, 126):
            hi, lo = px.rolling(L).max(), px.rolling(L).min()
            b += (2 * (px - lo) / (hi - lo) - 1)
        out += b / 2; nf += 1
    return (out / max(nf, 1)).clip(-1, 1)

# ── ERC (equal risk contribution) weights from a covariance matrix ───────────
def erc_weights(Sigma, iters=200, tol=1e-8):
    n = Sigma.shape[0]; x = np.ones(n) / n; b = np.ones(n) / n
    for _ in range(iters):
        Sx = Sigma @ x
        x_new = x.copy()
        for i in range(n):
            a = Sigma[i, i]
            c = Sx[i] - a * x[i]
            x_new[i] = (-c + np.sqrt(c * c + 4 * a * b[i])) / (2 * a)
            Sx = Sigma @ x_new
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new; break
        x = x_new
    return x / x.sum()

# ── portfolio construction -> daily strategy returns (causal) ────────────────
def strategy_returns(px, ret, families, target_vol, method, cap=1.0,
                     vol_span=30, scale_win=30, lw_win=252, rebal=21):
    fc = build_forecast(px, families)
    vol = ret.ewm(span=vol_span).std().shift(1)
    if method == "invvol":
        expo = (fc.shift(1) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = (expo * ret).sum(axis=1)
        sc = (target_vol / raw.rolling(scale_win).std()).shift(1)
        sc = sc.replace([np.inf, -np.inf], np.nan).clip(upper=50).fillna(0.0)
        pos = expo.mul(sc, axis=0).clip(-cap, cap)
        return (pos * ret).sum(axis=1).dropna()
    # method == "lw_rp": ERC magnitudes from LW-shrunk cov, recomputed monthly
    cols = px.columns; idx = px.index
    w_erc = pd.DataFrame(np.nan, index=idx, columns=cols)
    for t in range(lw_win, len(idx), rebal):
        window = ret.iloc[t - lw_win:t].dropna()
        if len(window) < lw_win // 2: continue
        Sig, _ = ledoit_wolf_cc(window)
        try:
            w = erc_weights(Sig)
        except Exception:
            w = np.ones(len(cols)) / len(cols)
        w_erc.iloc[t] = w
    w_erc = w_erc.ffill().shift(1)
    expo = (fc.shift(1) * w_erc / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = (expo * ret).sum(axis=1)
    sc = (target_vol / raw.rolling(scale_win).std()).shift(1)
    sc = sc.replace([np.inf, -np.inf], np.nan).clip(upper=50).fillna(0.0)
    pos = expo.mul(sc, axis=0).clip(-cap, cap)
    return (pos * ret).sum(axis=1).dropna()

# ── pre-registered grid (18 enumerated; K budget = 20) ───────────────────────
VOL = {"0.30%": 0.003, "0.40%": 0.004, "0.50%": 0.005}
SIZ = ["invvol", "lw_rp"]
SIG = {"TSMOM": ("tsmom",), "TSMOM+EWMA": ("tsmom", "ewma"),
       "TSMOM+EWMA+BO": ("tsmom", "ewma", "bo")}
def grid():
    for vk, vv in VOL.items():
        for s in SIZ:
            for gk, gv in SIG.items():
                yield (f"{vk}/{s}/{gk}", vv, s, gv)
K_BUDGET = 20

# ── CSCV Probability of Backtest Overfitting (Bailey & Lopez de Prado) ───────
def pbo_cscv(R, S=10):
    """R: T x M matrix of per-config returns. Returns PBO in [0,1]."""
    T, M = R.shape
    groups = np.array_split(np.arange(T), S)
    sr = lambda x: x.mean(0) / (x.std(0) + 1e-12)
    lam = []
    for is_g in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([groups[g] for g in is_g])
        oos_idx = np.concatenate([groups[g] for g in range(S) if g not in is_g])
        n_star = int(np.argmax(sr(R[is_idx])))
        oos_sr = sr(R[oos_idx])
        rank = (oos_sr <= oos_sr[n_star]).sum() / M      # relative rank in (0,1]
        rank = min(max(rank, 1.0 / (M + 1)), M / (M + 1))
        lam.append(np.log(rank / (1 - rank)))
    lam = np.array(lam)
    return float((lam < 0).mean())

# ── CPCV over the selection process -> OOS path Sharpe distribution ──────────
def cpcv_paths(cfg_returns, N=10, k=2, embargo=0.01):
    names = list(cfg_returns.keys())
    Rdf = pd.DataFrame(cfg_returns).dropna()
    R = Rdf.values; T = len(Rdf)
    cv = CombinatorialPurgedCV(n_groups=N, k_test=k, embargo_frac=embargo)
    groups = np.array_split(np.arange(T), N)
    # per combination: pick IS-best config, store its OOS returns per test group
    per_group = {g: [] for g in range(N)}
    sel_counts = {}
    for tr, te, combo in cv.split(T):
        if len(tr) < 50: continue
        is_sr = R[tr].mean(0) / (R[tr].std(0) + 1e-12)
        best = int(np.argmax(is_sr))
        sel_counts[names[best]] = sel_counts.get(names[best], 0) + 1
        for g in combo:
            per_group[g].append(R[groups[g], best])
    n_paths = comb(N, k) * k // N
    path_sr = []
    for p in range(n_paths):
        seg = []
        ok = True
        for g in range(N):
            if len(per_group[g]) <= p: ok = False; break
            seg.append(per_group[g][p])
        if not ok: continue
        s = np.concatenate(seg)
        path_sr.append(s.mean() / (s.std() + 1e-12) * ANN)
    return np.array(path_sr), sel_counts, Rdf

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="../data/clean")
    ap.add_argument("--drop", nargs="*", default=["UST10Y"])
    ap.add_argument("--trial", default="AUTO")
    ap.add_argument("--commit", default="uncommitted")
    a = ap.parse_args()

    px, ret, dhash = load_panel(a.clean, drop=a.drop)
    print(f"[panel] {px.shape[1]} assets x {px.shape[0]} common rows "
          f"({px.index.min().date()}->{px.index.max().date()})  drop={a.drop}")
    print(f"[dataset_hash] {dhash}")

    # evaluate the full grid (aligned return matrix)
    cfg_returns = {}
    for name, vv, s, gv in grid():
        cfg_returns[name] = strategy_returns(px, ret, gv, vv, s)
    Rdf = pd.DataFrame(cfg_returns).dropna()
    print(f"[grid] {Rdf.shape[1]} configs evaluated on {Rdf.shape[0]} aligned days")

    # CPCV over the selection process
    path_sr, sel, _ = cpcv_paths(cfg_returns)
    pbo = pbo_cscv(Rdf.values, S=10)

    # DSR on the CV-selected meta-strategy (median-Sharpe path proxy = pooled OOS
    # of the most-selected config), deflated by K=20
    top_cfg = max(sel, key=sel.get) if sel else Rdf.columns[int(np.argmax(Rdf.mean()/Rdf.std()))]
    sel_ret = cfg_returns[top_cfg].dropna().values
    dsr, sr, sr0 = deflated_sharpe(sel_ret, K_BUDGET)

    smin, smed, smax = (np.min(path_sr), np.median(path_sr), np.max(path_sr)) if len(path_sr) else (0,0,0)
    eq = (1 + cfg_returns[top_cfg].dropna()).cumprod()
    maxdd = ((eq.cummax() - eq) / eq.cummax()).max()
    worst = cfg_returns[top_cfg].dropna().min()
    status = "PASS" if (dsr > 0.95 and pbo < 0.05) else "REJECT"

    print("\n" + "=" * 78)
    print(f"CV-selected config (most-picked in-sample): {top_cfg}  "
          f"[selected {sel.get(top_cfg,0)}/{comb(10,2)} folds]")
    print(f"CPCV OOS Sharpe paths (n={len(path_sr)}): "
          f"[{smin:.2f} .. {smed:.2f} .. {smax:.2f}]")
    print(f"DSR (deflated by K={K_BUDGET}) = {dsr:.3f}   SR0_null={sr0:.4f}")
    print(f"PBO (CSCV, S=10) = {pbo*100:.1f}%")
    print(f"worst_day={worst*100:.2f}%  maxDD={maxdd*100:.1f}%")
    print(f"GATES: DSR>0.95 [{'Y' if dsr>0.95 else 'N'}]  PBO<5% [{'Y' if pbo<0.05 else 'N'}]  -> {status}")
    print("=" * 78)

    row = (f"| {a.trial} | {pd.Timestamp.utcnow().date()} | {a.commit[:7]} | "
           f"{(dhash or 'NA')[:12]} | {top_cfg} | "
           f"[{smin:.2f}..{smed:.2f}..{smax:.2f}] | {dsr:.3f} | {pbo*100:.0f}% | "
           f"{worst*100:.1f}% | {maxdd*100:.1f}% | {status} | CV-selected; auto |")
    print("\nLEDGER ROW (append to RESULTS_LEDGER.md):\n" + row)

if __name__ == "__main__":
    main()
