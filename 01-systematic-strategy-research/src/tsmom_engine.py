#!/usr/bin/env python3
"""
TSMOM ENGINE v0.2 — vol-targeted time-series momentum, basket of macro assets.
Upgrade over v0.1: naive 60/40 split REPLACED by Lopez de Prado Purged K-Fold CV
with embargo (AFML Ch.7, Snippets 7.1-7.3), plus Deflated Sharpe gate (Ch.12).
"""
import pandas as pd, numpy as np, os, itertools
from mathx import ncdf, nppf, skew, kurtosis

DATA="/sessions/festive-funny-gates/mnt/Mft/for claude/DATA FILE"
FILES={
 "US100":"((now))US100.cash_M5_202412310520_202606042345.csv",
 "US30":"US30.cash_M5_202412310445_202606042345.csv",
 "US500":"((now))US500.cash_M5_202412310920_202606042345.csv",
 "XAUUSD":"((now))XAUUSD_M5_202501020840_202606042345.csv",
 "EURUSD":"((now))EURUSD_M5_202501300620_202606050025.csv",
 "GBPUSD":"((now))GBPUSD_M5_202501272230_202606050025.csv",
}
PT={"US100":0.01,"US30":0.01,"US500":0.01,"XAUUSD":0.01,"EURUSD":0.00001,"GBPUSD":0.00001}

# ── data ──────────────────────────────────────────────────────────────────
def load_daily(sym):
    df=pd.read_csv(os.path.join(DATA,FILES[sym]),sep="\t");df.columns=[c.strip("<>").upper() for c in df.columns]
    df["TS"]=pd.to_datetime(df["DATE"]+" "+df["TIME"],format="%Y.%m.%d %H:%M:%S")
    df=df.set_index("TS")
    return df["CLOSE"].resample("1D").last().dropna(),(df["SPREAD"]*PT[sym]/df["CLOSE"]).resample("1D").mean()
def build():
    cl={};co={}
    for s in FILES: cl[s],co[s]=load_daily(s)
    px=pd.DataFrame(cl).dropna(); cost=pd.DataFrame(co).reindex(px.index).ffill()
    return px,px.pct_change(),cost

# ── strategy (causal: all signals/vols lagged) ──────────────────────────────
def tsmom_returns(lookbacks=(20,60,120),vol_span=30,target_daily_vol=0.006,scale_win=30,cost_on=True):
    px,ret,cost=build(); syms=list(px.columns)
    sig=pd.DataFrame(0.0,index=px.index,columns=syms)
    for L in lookbacks: sig+=np.sign(px/px.shift(L)-1.0)
    sig/=len(lookbacks)
    vol=ret.ewm(span=vol_span).std().shift(1)
    expo=(sig.shift(1)/vol).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    raw=(expo*ret).sum(axis=1)
    scale=(target_daily_vol/raw.rolling(scale_win).std()).shift(1).replace([np.inf,-np.inf],np.nan).clip(upper=50).fillna(0.0)
    pos=expo.mul(scale,axis=0); gross=(pos*ret).sum(axis=1)
    fric=(pos.diff().abs()*cost).sum(axis=1) if cost_on else 0.0*gross
    return (gross-fric).dropna()

# ── Lopez de Prado Purged K-Fold CV  (AFML Snippet 7.3) ─────────────────────
class PurgedKFold:
    """K-Fold for time series with overlapping labels. Faithful to AFML 7.4.
       t1: pd.Series, index=obs time, value=time the obs's information concludes
           (label/holding horizon end). pct_embargo: fraction of T embargoed
           AFTER each test fold to kill serial-correlation leakage."""
    def __init__(self,n_splits=6,t1=None,pct_embargo=0.0):
        self.n_splits=n_splits;self.t1=t1;self.pct_embargo=pct_embargo
    def split(self,X):
        idx=np.arange(X.shape[0]); emb=int(X.shape[0]*self.pct_embargo)
        folds=[(f[0],f[-1]+1) for f in np.array_split(idx,self.n_splits)]
        for i,j in folds:
            t0=self.t1.index[i]                              # test start time
            test=idx[i:j]
            maxt1=self.t1.index.searchsorted(self.t1.iloc[test].max())
            # train = obs whose info concluded before test starts (purge overlaps)
            train=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            train=train[train<X.shape[0]]
            if maxt1<X.shape[0]:                             # + post-test, embargoed
                train=np.concatenate((train,idx[maxt1+emb:]))
            yield train,test

# ── Deflated Sharpe (AFML Ch.12) ────────────────────────────────────────────
def deflated_sharpe(R,K):
    R=np.asarray(R); sr=R.mean()/R.std(); g3=skew(R); g4=kurtosis(R,fisher=False); N=len(R)
    sd=np.sqrt((1-g3*sr+(g4-1)/4*sr**2)/(N-1)); gam=0.5772156649
    sr0=0.0 if K<2 else sd*((1-gam)*nppf(1-1/K)+gam*nppf(1-1/(K*np.e)))
    return ncdf((sr-sr0)/sd),sr,sr0

def ann(sr_daily): return sr_daily*np.sqrt(252)

# ── purged CV evaluation of the (rule-based) strategy ───────────────────────
def purged_cv_eval(net,max_lookback=120,n_splits=6):
    # label horizon: each day's signal depends on up to `max_lookback` prior days.
    # embargo must cover that serial dependence -> pct_embargo = max_lookback/T
    T=len(net); t1=pd.Series(net.index,index=net.index)   # same-day realization
    emb=max_lookback/T
    pk=PurgedKFold(n_splits=n_splits,t1=t1,pct_embargo=emb)
    fold_sr=[]
    for tr,te in pk.split(net.to_frame()):
        seg=net.iloc[te]
        if len(seg)>5 and seg.std()>0: fold_sr.append(ann(seg.mean()/seg.std()))
    return np.array(fold_sr),emb

if __name__=="__main__":
    print("="*72);print("TSMOM ENGINE v0.2 — PURGED K-FOLD CV (LdP AFML Ch.7) + DSR GATE")
    print("="*72)
    net=tsmom_returns()
    eq=(1+net).cumprod()
    print(f"obs={len(net)}  ann_ret={(eq.iloc[-1]**(252/len(net))-1)*100:.1f}%  "
          f"SR(ann)={ann(net.mean()/net.std()):.2f}  maxDD={((eq.cummax()-eq)/eq.cummax()).max()*100:.1f}%  "
          f"skew={skew(net.values):.2f}")
    fold_sr,emb=purged_cv_eval(net)
    print(f"\nPurged K-Fold (6 folds, embargo={emb*100:.1f}% of sample):")
    print(f"  per-fold OOS Sharpe(ann): {np.round(fold_sr,2).tolist()}")
    print(f"  mean OOS Sharpe={fold_sr.mean():.2f}  std={fold_sr.std():.2f}  "
          f"#folds positive={int((fold_sr>0).sum())}/{len(fold_sr)}")
    # DSR deflated by full parameter grid (honest selection-bias correction)
    LBs=[(20,60,120),(20,40),(60,120),(10,30,60)]; TVs=[0.004,0.006,0.008]
    K=len(list(itertools.product(LBs,TVs)))
    dsr,sr,sr0=deflated_sharpe(net.values,K)
    print(f"\nDSR GATE (deflated by K={K} configs): DSR={dsr:.3f}  [PASS needs >0.95]")
    print(f"  verdict: {'PASS' if dsr>0.95 else 'REJECT — not distinguishable from selection-bias noise'}")
