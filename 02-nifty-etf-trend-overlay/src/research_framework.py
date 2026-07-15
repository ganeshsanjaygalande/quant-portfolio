#!/usr/bin/env python3
"""
RESEARCH FRAMEWORK — building blocks for the large-universe TSMOM study.
Implements the three upgrades from the master plan:
  (A) Ledoit-Wolf constant-correlation covariance shrinkage  (LW 2004)
  (B) Signal ensembling  (multi-lookback TSMOM + EWMA crossover + breakout)
  (C) Combinatorial Purged Cross-Validation  (Lopez de Prado AFML Ch.12)
Pure numpy/pandas (no sklearn/scipy) so it runs anywhere.
"""
import numpy as np, pandas as pd, itertools, os
from mathx import ncdf, nppf, skew, kurtosis

# ── (A) Ledoit-Wolf constant-correlation shrinkage ──────────────────────────
def ledoit_wolf_cc(returns):
    """returns: T x N DataFrame of (excess) returns. -> (Sigma_shrunk, delta)."""
    X = returns.values
    X = X - X.mean(0)
    T, N = X.shape
    S = (X.T @ X) / T                       # sample covariance
    var = np.diag(S)
    std = np.sqrt(var)
    corr = S / np.outer(std, std)
    rbar = (corr.sum() - N) / (N * (N - 1)) # avg off-diag correlation
    F = rbar * np.outer(std, std)           # constant-correlation target
    np.fill_diagonal(F, var)
    # pi: sum of asymptotic variances of sample cov entries
    Xsq = X ** 2
    pi_mat = (Xsq.T @ Xsq) / T - S ** 2
    pi_hat = pi_mat.sum()
    # rho: sum of asymptotic covariances (diagonal + off-diagonal term)
    rho_diag = np.diag(pi_mat).sum()
    term = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            t1 = ((X[:, i] ** 2 - S[i, i]) * (X[:, i] * X[:, j] - S[i, j])).mean()
            t2 = ((X[:, j] ** 2 - S[j, j]) * (X[:, i] * X[:, j] - S[i, j])).mean()
            term[i, j] = (std[j] / std[i]) * t1 + (std[i] / std[j]) * t2
    rho_hat = rho_diag + (rbar / 2.0) * term.sum()
    gamma_hat = ((F - S) ** 2).sum()
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / T))
    Sigma = delta * F + (1 - delta) * S
    return Sigma, delta

# ── (B) Signal ensembling ───────────────────────────────────────────────────
def ensemble_forecast(px, tsmom_lb=(21,63,126,252), ewma_pairs=((8,24),(16,48),(32,96)),
                      breakout_lb=(63,126)):
    """Blend TSMOM sign, EWMA crossover, and Donchian breakout into [-1,1]/asset.
       Each family contributes equally; within a family, lookbacks are averaged."""
    out = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    fams = 0
    # 1. time-series momentum: sign of trailing return
    m = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for L in tsmom_lb: m += np.sign(px / px.shift(L) - 1)
    out += (m / len(tsmom_lb)); fams += 1
    # 2. EWMA crossover (fast-slow), normalized by its own rolling vol
    c = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for f, s in ewma_pairs:
        x = px.ewm(span=f).mean() - px.ewm(span=s).mean()
        c += np.tanh(x / x.rolling(252, min_periods=63).std())
    out += (c / len(ewma_pairs)); fams += 1
    # 3. Donchian breakout: +1 near high, -1 near low
    b = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for L in breakout_lb:
        hi = px.rolling(L).max(); lo = px.rolling(L).min()
        b += (2 * (px - lo) / (hi - lo) - 1)
    out += (b / len(breakout_lb)); fams += 1
    return (out / fams).clip(-1, 1)

# ── (C) Combinatorial Purged CV (LdP Ch.12) ─────────────────────────────────
class CombinatorialPurgedCV:
    """Partition T obs into N groups; test every k-subset of groups (C(N,k) splits),
       purging+embargoing train obs adjacent to each test group. Yields a
       DISTRIBUTION of backtest paths, not one split. phi_paths = C(N,k)*k/N."""
    def __init__(self, n_groups=6, k_test=2, embargo_frac=0.01):
        self.N, self.k, self.emb = n_groups, k_test, embargo_frac
    def n_paths(self):
        from math import comb
        return comb(self.N, self.k) * self.k // self.N
    def split(self, T):
        idx = np.arange(T); groups = np.array_split(idx, self.N)
        emb = int(T * self.emb)
        for test_combo in itertools.combinations(range(self.N), self.k):
            test_idx = np.concatenate([groups[g] for g in test_combo])
            test_set = set(test_idx.tolist())
            train = []
            for g in range(self.N):
                if g in test_combo: continue
                gi = groups[g]
                # purge: drop group obs adjacent (within embargo) to any test group
                lo, hi = gi[0], gi[-1]
                touch = any((min(groups[tc][-1], hi) >= max(groups[tc][0], lo) - emb and
                             min(groups[tc][-1]+emb, hi) >= max(groups[tc][0]-emb, lo))
                            for tc in test_combo)
                keep = gi
                # embargo obs immediately after any test block
                keep = np.array([x for x in keep if not any(
                    tcg[-1] < x <= tcg[-1] + emb for tcg in [groups[t] for t in test_combo])])
                train.append(keep)
            train_idx = np.concatenate(train) if train else np.array([], int)
            yield np.sort(train_idx), np.sort(test_idx), test_combo

def deflated_sharpe(R, K, sr_var=None):
    R = np.asarray(R); sr = R.mean()/R.std(); g3 = skew(R); g4 = kurtosis(R, fisher=False); N=len(R)
    sd = np.sqrt((1 - g3*sr + (g4-1)/4*sr**2)/(N-1)) if sr_var is None else np.sqrt(sr_var)
    gam = 0.5772156649
    sr0 = 0 if K < 2 else sd*((1-gam)*nppf(1-1/K) + gam*nppf(1-1/(K*np.e)))
    return ncdf((sr - sr0)/sd), sr, sr0

# ── smoke test on the 6-asset macro data (proves the code runs; NOT a result) ─
if __name__ == "__main__":
    MD = "/sessions/festive-funny-gates/mnt/Mft/for claude/macro_data"
    def ld(p):
        f=[x for x in os.listdir(MD) if x.startswith(p) and "_D1_" in x][0]
        d=pd.read_csv(os.path.join(MD,f),sep="\t");d.columns=[c.strip("<>").upper() for c in d.columns]
        d["TS"]=pd.to_datetime(d["DATE"]+" "+d["TIME"],format="%Y.%m.%d %H:%M:%S")
        return d.set_index("TS")["CLOSE"]
    S={"US100":"US100_D1","XAUUSD":"XAUUSD_D1","USOIL":"USOIL_D1","BTCUSD":"BTCUSD_D1","EURUSD":"EURUSD_D1","TLT":"TLT.NAS_D1"}
    px=pd.DataFrame({k:ld(v).rename(k) for k,v in S.items()}); px.index=px.index.normalize()
    px=px[~px.index.duplicated()].dropna().sort_index(); ret=px.pct_change().dropna()
    print("SMOKE TEST (6-asset stand-in; framework validation only)")
    Sig, delta = ledoit_wolf_cc(ret)
    print(f"(A) Ledoit-Wolf: delta={delta:.3f}  PSD={np.all(np.linalg.eigvals(Sig)>-1e-10)}  condSample={np.linalg.cond(ret.cov().values):.1f} condShrunk={np.linalg.cond(Sig):.1f}")
    F = ensemble_forecast(px)
    print(f"(B) Ensemble forecast: shape={F.shape} range=[{F.min().min():.2f},{F.max().max():.2f}] last={F.iloc[-1].round(2).to_dict()}")
    cv = CombinatorialPurgedCV(n_groups=6, k_test=2, embargo_frac=0.01)
    splits=list(cv.split(len(ret)))
    print(f"(C) CPCV: N=6 k=2 -> {len(splits)} splits, {cv.n_paths()} backtest paths; "
          f"e.g. split0 train={len(splits[0][0])} test={len(splits[0][1])} testgroups={splits[0][2]}")
