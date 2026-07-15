import pandas as pd, numpy as np, os, itertools
from mathx import ncdf, nppf, skew, kurtosis
MD="/sessions/festive-funny-gates/mnt/Mft/for claude/macro_data"
SYMS=["US100","US500","XAUUSD","USOIL","BTCUSD","EURUSD"]
def load_d1(s):
    f=[x for x in os.listdir(MD) if x.startswith(s+"_D1_")][0]
    df=pd.read_csv(os.path.join(MD,f),sep="\t");df.columns=[c.strip("<>").upper() for c in df.columns]
    df["TS"]=pd.to_datetime(df["DATE"]+" "+df["TIME"],format="%Y.%m.%d %H:%M:%S")
    return df.set_index("TS")["CLOSE"].rename(s)
# ── QC ──
print("=== DATA QC ===")
series={}
for s in SYMS:
    c=load_d1(s); series[s]=c
    rets=c.pct_change()
    dup=c.index.duplicated().sum(); zeros=(c<=0).sum(); nan=c.isna().sum()
    flat=(rets==0).sum()
    print(f"{s:7s} n={len(c)} {c.index.min().date()}->{c.index.max().date()} "
          f"dup={dup} zero/neg={zeros} nan={nan} flat0ret={flat} "
          f"maxabsret={rets.abs().max()*100:.1f}%")
# D1: timezone offset is immaterial to date alignment; inner-join on date avoids ffill weekend artifacts
px=pd.DataFrame(series)
# normalize index to date (drop intraday hh:mm that's all 00:00:00 anyway)
px.index=px.index.normalize()
px=px[~px.index.duplicated(keep='last')]
print(f"\nraw union dates={len(px)}  all-present(inner)={len(px.dropna())}")
# Per instruction: build unified index + ffill; then require all cols valid (drop leading NaN)
px_ff=px.sort_index().ffill().dropna()
print(f"after ffill+dropna leading: {len(px_ff)} rows  {px_ff.index.min().date()}->{px_ff.index.max().date()}")
ret=px_ff.pct_change()
print("\n=== correlation of daily returns (diversification check) ===")
print(ret.corr().round(2).to_string())

# ── TSMOM (lookbacks capped so embargo<5%) ──
def tsmom(lookbacks=(20,40,60),vol_span=30,tgt=0.006,scale_win=30):
    sig=pd.DataFrame(0.0,index=px_ff.index,columns=px_ff.columns)
    for L in lookbacks: sig+=np.sign(px_ff/px_ff.shift(L)-1)
    sig/=len(lookbacks)
    vol=ret.ewm(span=vol_span).std().shift(1)
    expo=(sig.shift(1)/vol).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    raw=(expo*ret).sum(axis=1)
    sc=(tgt/raw.rolling(scale_win).std()).shift(1).replace([np.inf,-np.inf],np.nan).clip(upper=50).fillna(0)
    return ((expo.mul(sc,axis=0))*ret).sum(axis=1).dropna()

class PurgedKFold:
    def __init__(s,n,t1,emb): s.n=n;s.t1=t1;s.emb=emb
    def split(s,X):
        idx=np.arange(len(X)); e=int(len(X)*s.emb)
        for i,j in [(f[0],f[-1]+1) for f in np.array_split(idx,s.n)]:
            t0=s.t1.index[i]; te=idx[i:j]
            mx=s.t1.index.searchsorted(s.t1.iloc[te].max())
            tr=s.t1.index.searchsorted(s.t1[s.t1<=t0].index); tr=tr[tr<len(X)]
            if mx<len(X): tr=np.concatenate((tr,idx[mx+e:]))
            yield tr,te

def dsr(R,K):
    R=np.asarray(R);sr=R.mean()/R.std();g3=skew(R);g4=kurtosis(R,fisher=False);N=len(R)
    sd=np.sqrt((1-g3*sr+(g4-1)/4*sr**2)/(N-1));gam=0.5772156649
    sr0=0 if K<2 else sd*((1-gam)*nppf(1-1/K)+gam*nppf(1-1/(K*np.e)))
    return ncdf((sr-sr0)/sd),sr,sr0
ann=lambda x:x*np.sqrt(252)

net=tsmom()
eq=(1+net).cumprod()
maxlb=60; emb=maxlb/len(net)
print(f"\n=== TSMOM MACRO (6 assets, lookbacks 20/40/60, embargo={emb*100:.1f}%) ===")
print(f"obs={len(net)} ann_ret={(eq.iloc[-1]**(252/len(net))-1)*100:.1f}% SR(ann)={ann(net.mean()/net.std()):.2f} "
      f"maxDD={((eq.cummax()-eq)/eq.cummax()).max()*100:.1f}% skew={skew(net.values):.2f} worstday={net.min()*100:.2f}%")
t1=pd.Series(net.index,index=net.index)
folds=[ann(net.iloc[te].mean()/net.iloc[te].std()) for tr,te in PurgedKFold(6,t1,emb).split(net.to_frame()) if net.iloc[te].std()>0]
folds=np.array(folds)
print(f"Purged 6-fold OOS Sharpe: {np.round(folds,2).tolist()}  mean={folds.mean():.2f} std={folds.std():.2f}")
LBs=[(20,40,60),(20,60),(10,30),(40,60)];TVs=[0.004,0.006,0.008];K=len(LBs)*len(TVs)
# pick best config by full-sample SR then deflate
best=max(itertools.product(LBs,TVs),key=lambda c:tsmom(c[0],tgt=c[1]).pipe(lambda r:r.mean()/r.std()))
bnet=tsmom(best[0],tgt=best[1])
d,sr,sr0=dsr(bnet.values,K)
print(f"DSR GATE: best={best} SR(ann)={ann(bnet.mean()/bnet.std()):.2f}  DSR(K={K})={d:.3f}  "
      f"-> {'PASS' if d>0.95 else 'REJECT'}")

print("\n"+"#"*64)
print("REFEREE RE-RUN: ffill injects BTC-weekend fake bars -> use clean inner-join")
print("#"*64)
px_clean=px.dropna().sort_index()   # only true common trading days, NO ffill
print(f"clean common-trading-days={len(px_clean)} (vs ffill {len(px_ff)}); "
      f"ffill added {len(px_ff)-len(px_clean)} fake weekend bars")
ret_c=px_clean.pct_change()
def tsmom_c(lookbacks=(20,40,60),vol_span=30,tgt=0.006,scale_win=30):
    sig=pd.DataFrame(0.0,index=px_clean.index,columns=px_clean.columns)
    for L in lookbacks: sig+=np.sign(px_clean/px_clean.shift(L)-1)
    sig/=len(lookbacks)
    vol=ret_c.ewm(span=vol_span).std().shift(1)
    expo=(sig.shift(1)/vol).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    raw=(expo*ret_c).sum(axis=1)
    sc=(tgt/raw.rolling(scale_win).std()).shift(1).replace([np.inf,-np.inf],np.nan).clip(upper=50).fillna(0)
    return ((expo.mul(sc,axis=0))*ret_c).sum(axis=1).dropna()
netc=tsmom_c(); eqc=(1+netc).cumprod(); emb=60/len(netc)
print(f"CLEAN TSMOM: obs={len(netc)} ann_ret={(eqc.iloc[-1]**(252/len(netc))-1)*100:.1f}% "
      f"SR(ann)={ann(netc.mean()/netc.std()):.2f} maxDD={((eqc.cummax()-eqc)/eqc.cummax()).max()*100:.1f}% "
      f"skew={skew(netc.values):.2f} worstday={netc.min()*100:.2f}% embargo={emb*100:.1f}%")
t1c=pd.Series(netc.index,index=netc.index)
fc=np.array([ann(netc.iloc[te].mean()/netc.iloc[te].std()) for tr,te in PurgedKFold(6,t1c,emb).split(netc.to_frame()) if netc.iloc[te].std()>0])
print(f"Purged 6-fold OOS Sharpe: {np.round(fc,2).tolist()} mean={fc.mean():.2f} std={fc.std():.2f} pos={int((fc>0).sum())}/{len(fc)}")
bestc=max(itertools.product(LBs,TVs),key=lambda c:tsmom_c(c[0],tgt=c[1]).pipe(lambda r:r.mean()/r.std()))
bnetc=tsmom_c(bestc[0],tgt=bestc[1]); dc,src,sr0c=dsr(bnetc.values,K)
print(f"DSR GATE (clean): best={bestc} SR(ann)={ann(bnetc.mean()/bnetc.std()):.2f} DSR(K={K})={dc:.3f} -> {'PASS' if dc>0.95 else 'REJECT'}")
# also DSR on the default (non-cherry-picked) config
dd,_,_=dsr(netc.values,K)
print(f"DSR on DEFAULT config (20/40/60, tgt .006): {dd:.3f} -> {'PASS' if dd>0.95 else 'REJECT'}")
