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
# relative round-trip spread cost (fraction of notional) from data point sizes
PT={"US100":0.01,"US30":0.01,"US500":0.01,"XAUUSD":0.01,"EURUSD":0.00001,"GBPUSD":0.00001}
def load_daily(sym):
    df=pd.read_csv(os.path.join(DATA,FILES[sym]),sep="\t");df.columns=[c.strip("<>").upper() for c in df.columns]
    df["TS"]=pd.to_datetime(df["DATE"]+" "+df["TIME"],format="%Y.%m.%d %H:%M:%S")
    df=df.set_index("TS")
    close=df["CLOSE"].resample("1D").last().dropna()
    relsp=(df["SPREAD"]*PT[sym]/df["CLOSE"]).resample("1D").mean()
    return close, relsp
def build():
    closes={};costs={}
    for s in FILES: closes[s],costs[s]=load_daily(s)
    px=pd.DataFrame(closes).dropna()
    cost=pd.DataFrame(costs).reindex(px.index).ffill()
    ret=px.pct_change()
    return px,ret,cost
def tsmom_backtest(lookbacks=(20,60,120),vol_span=30,target_daily_vol=0.006,
                   scale_win=30,cost_on=True):
    px,ret,cost=build()
    syms=list(px.columns); n=len(px)
    # signal: average sign of trailing-L return across lookbacks
    sig=pd.DataFrame(0.0,index=px.index,columns=syms)
    for L in lookbacks:
        sig+=np.sign(px/px.shift(L)-1.0)
    sig/=len(lookbacks)
    # per-instrument daily vol (EWMA), lagged
    vol=ret.ewm(span=vol_span).std().shift(1)
    # raw inverse-vol signed exposure (fraction of equity), lagged signal
    expo=(sig.shift(1)/vol).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    raw_ret=(expo*ret).sum(axis=1)  # unscaled portfolio return
    # scale to target portfolio vol using trailing realized vol (lagged)
    scale=(target_daily_vol/raw_ret.rolling(scale_win).std()).shift(1).replace([np.inf,-np.inf],np.nan)
    scale=scale.clip(upper=50).fillna(0.0)
    pos=expo.mul(scale,axis=0)              # final fractional exposures
    gross=(pos*ret).sum(axis=1)
    # friction: turnover * relative spread (round trip ~ full bid-ask on the changed exposure)
    turn=pos.diff().abs()
    fric=(turn*cost).sum(axis=1) if cost_on else 0.0*gross
    net=gross-fric
    net=net.dropna()
    return net,pos
def metrics(net):
    n=len(net); mu=net.mean(); sd=net.std()
    sr_d=mu/sd if sd>0 else 0
    sr_ann=sr_d*np.sqrt(252)
    eq=(1+net).cumprod()
    dd=(eq.cummax()-eq)/eq.cummax()
    # worst single-day loss (FTMO daily DD proxy)
    return dict(n=n,mu_d=mu,sd_d=sd,sr_ann=sr_ann,
        tot_ret=eq.iloc[-1]-1,maxdd=dd.max(),worst_day=net.min(),
        days_to_target=None,skew=skew(net.values),kurt=kurtosis(net.values),
        ann_ret=(eq.iloc[-1]**(252/n)-1))
def dsr(net,K):
    R=net.values; sr=R.mean()/R.std(); g3=skew(R); g4=kurtosis(R,fisher=False); N=len(R)
    var=(1-g3*sr+(g4-1)/4*sr**2)/(N-1); sd=np.sqrt(var); gam=0.5772156649
    if K<2: sr0=0.0
    else:
        sr0=sd*((1-gam)*nppf(1-1/K)+gam*nppf(1-1/(K*np.e)))
    return ncdf((sr-sr0)/sd), sr, sr0

print("="*72);print("TSMOM v0.1 PROTOTYPE — vol-targeted, basket of 6, daily")
print("="*72)
net,pos=tsmom_backtest()
m=metrics(net)
print(f"obs days={m['n']}  ann_ret={m['ann_ret']*100:.1f}%  tot_ret={m['tot_ret']*100:.1f}%")
print(f"SR(ann)={m['sr_ann']:.2f}  maxDD={m['maxdd']*100:.1f}%  worst_day={m['worst_day']*100:.2f}%  skew={m['skew']:.2f}")
print(f"daily vol realized={m['sd_d']*100:.2f}% (target 0.60%)")

print("\n--- FTMO feasibility (start $50k, target +10%, daily<5%, total<10%) ---")
eq=(1+net).cumprod()*50000
breaches_daily=(net< -0.05).sum()
peak=eq.cummax(); tot_dd=(peak-eq)/peak
print(f"days with >5% single-day loss: {breaches_daily}")
print(f"max total drawdown: {tot_dd.max()*100:.1f}%  (>10% would fail)")
print(f"final equity: ${eq.iloc[-1]:,.0f}")

print("\n--- train/test split (60/40) ---")
half=int(len(net)*0.6)
for lab,seg in [("TRAIN",net.iloc[:half]),("TEST",net.iloc[half:])]:
    mm=metrics(seg);print(f"{lab}: SR(ann)={mm['sr_ann']:.2f} ann_ret={mm['ann_ret']*100:.1f}% maxDD={mm['maxdd']*100:.1f}%")

print("\n--- DSR GATE (deflated by parameter search) ---")
# fair grid
LBs=[(20,60,120),(20,40),(60,120),(10,30,60)]; TVs=[0.004,0.006,0.008]
configs=list(itertools.product(LBs,TVs)); K=len(configs)
best=None
for lb,tv in configs:
    nt,_=tsmom_backtest(lookbacks=lb,target_daily_vol=tv)
    mm=metrics(nt)
    if best is None or mm['sr_ann']>best[2]: best=(lb,tv,mm['sr_ann'],nt)
d,sr,sr0=dsr(best[3],K)
print(f"configs tried K={K}; best={best[0]} tv={best[1]} SR(ann)={best[2]:.2f}")
print(f"DSR (deflated by K={K}) = {d:.3f}  [need >0.95]   sr0_null={sr0:.4f}")

print("\n--- per-instrument standalone TSMOM contribution (SR ann) ---")
px,ret,cost=build()
for s in px.columns:
    sub_net,_=tsmom_backtest()
# quick: contribution via single-asset vol-target sign strategy
for s in px.columns:
    r=ret[s]; sg=np.sign(px[s]/px[s].shift(60)-1).shift(1)
    v=r.ewm(span=30).std().shift(1); e=(sg/v).fillna(0)
    pr=(e*r); pr=pr/pr.std()*0.006  # rough scale
    pr=pr.dropna()
    print(f"  {s}: SR(ann)={pr.mean()/pr.std()*np.sqrt(252):.2f}")
