import pandas as pd, numpy as np, os, itertools
from mathx import ncdf, nppf, skew, kurtosis
MD="/sessions/festive-funny-gates/mnt/Mft/for claude/macro_data"
def load_d1(prefix):
    f=[x for x in os.listdir(MD) if x.startswith(prefix) and "_D1_" in x][0]
    df=pd.read_csv(os.path.join(MD,f),sep="\t");df.columns=[c.strip("<>").upper() for c in df.columns]
    df["TS"]=pd.to_datetime(df["DATE"]+" "+df["TIME"],format="%Y.%m.%d %H:%M:%S")
    return df.set_index("TS")["CLOSE"]
# Report#2 revision: drop US500, add TLT (rates). UST10Y excluded (only 17mo, single contract).
SET={"US100":"US100_D1","XAUUSD":"XAUUSD_D1","USOIL":"USOIL_D1",
     "BTCUSD":"BTCUSD_D1","EURUSD":"EURUSD_D1","TLT":"TLT.NAS_D1"}
series={k:load_d1(v).rename(k) for k,v in SET.items()}
px=pd.DataFrame(series); px.index=px.index.normalize(); px=px[~px.index.duplicated(keep='last')]
px=px.dropna().sort_index()              # clean inner-join, NO ffill
ret=px.pct_change()
print(f"assets={list(px.columns)}  common trading days={len(px)}  {px.index.min().date()}->{px.index.max().date()}")
print("\n=== daily-return correlation (TLT = the new diversifier) ===")
print(ret.corr().round(2).to_string())

EXPO_CAP=1.0   # per-instrument |notional| <= 1.0x equity (jump cap)
def tsmom(lookbacks=(20,40,60),vol_span=30,tgt=0.004,scale_win=30,cap=EXPO_CAP):
    sig=pd.DataFrame(0.0,index=px.index,columns=px.columns)
    for L in lookbacks: sig+=np.sign(px/px.shift(L)-1)
    sig/=len(lookbacks)
    vol=ret.ewm(span=vol_span).std().shift(1)
    expo=(sig.shift(1)/vol).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    raw=(expo*ret).sum(axis=1)
    sc=(tgt/raw.rolling(scale_win).std()).shift(1).replace([np.inf,-np.inf],np.nan).clip(upper=50).fillna(0)
    pos=expo.mul(sc,axis=0)
    pos=pos.clip(lower=-cap,upper=cap)   # per-instrument jump cap
    return (pos*ret).sum(axis=1).dropna()

class PurgedKFold:
    def __init__(s,n,t1,emb): s.n=n;s.t1=t1;s.emb=emb
    def split(s,X):
        idx=np.arange(len(X)); e=int(len(X)*s.emb)
        for i,j in [(f[0],f[-1]+1) for f in np.array_split(idx,s.n)]:
            t0=s.t1.index[i]; te=idx[i:j]; mx=s.t1.index.searchsorted(s.t1.iloc[te].max())
            tr=s.t1.index.searchsorted(s.t1[s.t1<=t0].index); tr=tr[tr<len(X)]
            if mx<len(X): tr=np.concatenate((tr,idx[mx+e:]))
            yield tr,te
def dsr(R,K):
    R=np.asarray(R);sr=R.mean()/R.std();g3=skew(R);g4=kurtosis(R,fisher=False);N=len(R)
    sd=np.sqrt((1-g3*sr+(g4-1)/4*sr**2)/(N-1));gam=0.5772156649
    sr0=0 if K<2 else sd*((1-gam)*nppf(1-1/K)+gam*nppf(1-1/(K*np.e)))
    return ncdf((sr-sr0)/sd),sr,sr0
ann=lambda x:x*np.sqrt(252)

net=tsmom(); eq=(1+net).cumprod(); emb=60/len(net)
print(f"\n=== TSMOM v3 (drop US500 + add TLT, tgt 0.4%, jump cap {EXPO_CAP}x) ===")
print(f"obs={len(net)} ann_ret={(eq.iloc[-1]**(252/len(net))-1)*100:.1f}% SR(ann)={ann(net.mean()/net.std()):.2f} "
      f"maxDD={((eq.cummax()-eq)/eq.cummax()).max()*100:.1f}% skew={skew(net.values):.2f} worstday={net.min()*100:.2f}%")
t1=pd.Series(net.index,index=net.index)
fc=np.array([ann(net.iloc[te].mean()/net.iloc[te].std()) for tr,te in PurgedKFold(6,t1,emb).split(net.to_frame()) if net.iloc[te].std()>0])
print(f"Purged 6-fold OOS Sharpe: {np.round(fc,2).tolist()} mean={fc.mean():.2f} std={fc.std():.2f} pos={int((fc>0).sum())}/{len(fc)}")
LBs=[(20,40,60),(20,60),(10,30),(40,60)];TVs=[0.003,0.004,0.006];K=len(LBs)*len(TVs)
best=max(itertools.product(LBs,TVs),key=lambda c:(lambda r:r.mean()/r.std())(tsmom(c[0],tgt=c[1])))
bnet=tsmom(best[0],tgt=best[1]); d,sr,sr0=dsr(bnet.values,K)
dd,_,_=dsr(net.values,K)
print(f"\nDSR GATE: best={best} SR(ann)={ann(bnet.mean()/bnet.std()):.2f} DSR(K={K})={d:.3f} -> {'PASS' if d>0.95 else 'REJECT'}")
print(f"DSR default(20/40/60,0.4%): {dd:.3f} -> {'PASS' if dd>0.95 else 'REJECT'}")
print(f"worst day (best cfg): {bnet.min()*100:.2f}%  maxDD(best): {((1+bnet).cumprod().cummax()-(1+bnet).cumprod()).div((1+bnet).cumprod().cummax()).max()*100:.1f}%")
