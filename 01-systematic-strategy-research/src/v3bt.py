import pandas as pd, numpy as np, os
from mathx import erfi, ncdf, nppf, skew, kurtosis, minimize_grid
DATA="/sessions/festive-funny-gates/mnt/Mft/for claude/DATA FILE"
NQF="((now))US100.cash_M5_202412310520_202606042345.csv"
YMF="US30.cash_M5_202412310445_202606042345.csv"
POINT=0.01  # index point size for SPREAD col
def load(f):
    df=pd.read_csv(os.path.join(DATA,f),sep="\t");df.columns=[c.strip("<>").upper() for c in df.columns]
    df["TS"]=pd.to_datetime(df["DATE"]+" "+df["TIME"],format="%Y.%m.%d %H:%M:%S")
    return df.set_index("TS")
def merged(tf="5min"):
    nq=load(NQF);ym=load(YMF)
    d=pd.DataFrame({"NQ":nq["CLOSE"],"NQ_H":nq["HIGH"],"NQ_L":nq["LOW"],"NQ_SP":nq["SPREAD"],
                    "YM":ym["CLOSE"]}).dropna()
    if tf!="5min":
        d=pd.DataFrame({
            "NQ":d["NQ"].resample(tf).last(),"NQ_H":d["NQ_H"].resample(tf).max(),
            "NQ_L":d["NQ_L"].resample(tf).min(),"NQ_SP":d["NQ_SP"].resample(tf).mean(),
            "YM":d["YM"].resample(tf).last()}).dropna()
    return d
def backtest(tf="5min",w=60,z_entry=2.0,z_stop=3.0,atr_mult=3.5,ema_span=200,atr_p=14,use_ema=True):
    d=merged(tf).copy()
    ratio=(d["NQ"]/d["YM"]).rolling(w).mean()
    spread=d["NQ"]-ratio*d["YM"]
    z=(spread-spread.rolling(w).mean())/spread.rolling(w).std()
    ema=d["NQ"].ewm(span=ema_span,adjust=False).mean()
    atr=(d["NQ_H"]-d["NQ_L"]).rolling(atr_p).mean()
    NQ=d["NQ"].values;H=d["NQ_H"].values;L=d["NQ_L"].values;SP=d["NQ_SP"].values*POINT
    Z=z.values;EMA=ema.values;ATR=atr.values;TS=d.index
    pos=0;ei=0;entry=sl=0.0;sgn=0;slpts=0.0
    trades=[]
    n=len(d)
    for i in range(max(w,ema_span)+1,n):
        if not(np.isfinite(Z[i]) and np.isfinite(ATR[i]) and ATR[i]>0): continue
        if pos==0:
            bull=NQ[i]>EMA[i] if use_ema else True
            bear=NQ[i]<EMA[i] if use_ema else True
            s=0
            if Z[i]<-z_entry and bull: s=1
            elif Z[i]>z_entry and bear: s=-1
            if s!=0:
                sgn=s;ei=i;entry=NQ[i];slpts=ATR[i]*atr_mult
                sl=entry-slpts if s==1 else entry+slpts
                pos=s
        else:
            exit_px=None;reason=None
            # broker SL intrabar
            if sgn==1 and L[i]<=sl: exit_px=sl;reason="SL"
            elif sgn==-1 and H[i]>=sl: exit_px=sl;reason="SL"
            elif abs(Z[i])>=z_stop: exit_px=NQ[i];reason="ALGO"
            elif (sgn==1 and Z[i]>=0) or (sgn==-1 and Z[i]<=0): exit_px=NQ[i];reason="TGT"
            if exit_px is not None:
                gross_pts=sgn*(exit_px-entry)
                fric_pts=SP[i]  # one full bid-ask round trip, single leg, $0 comm
                net_pts=gross_pts-fric_pts
                R=net_pts/slpts  # risk-multiple (each trade risks 1 R = slpts)
                trades.append(dict(ts=TS[i],bars=i-ei,reason=reason,sgn=sgn,
                    gross_pts=gross_pts,net_pts=net_pts,R=R,day=TS[i].date()))
                pos=0
    return pd.DataFrame(trades)
def stats(tr,tf,risk=0.015,bars_per_yr=None):
    if len(tr)==0: return {}
    R=tr["R"].values
    n=len(R)
    days=(tr["ts"].iloc[-1]-tr["ts"].iloc[0]).days or 1
    tpd=n/days
    cum_R=R.sum()
    eq=1.0+np.cumsum(R*risk)  # equity path in fraction of start
    dd=np.maximum.accumulate(eq)-eq
    sr_trade=R.mean()/R.std() if R.std()>0 else 0
    sr_ann=sr_trade*np.sqrt(n/ (days/365.0))  # annualized by trades/yr
    return dict(tf=tf,trades=n,win=(R>0).mean()*100,avgR=R.mean(),medR=np.median(R),
        sumR=cum_R,ret_pct=cum_R*risk*100,maxdd_pct=dd.max()*100,
        skew=skew(R),kurt=kurtosis(R),sr_trade=sr_trade,sr_ann=sr_ann,
        tpd=tpd,days=days,
        sl_pct=(tr["reason"]=="SL").mean()*100,tgt_pct=(tr["reason"]=="TGT").mean()*100,
        algo_pct=(tr["reason"]=="ALGO").mean()*100,longpct=(tr["sgn"]==1).mean()*100,R=R)
def deflated_sharpe(R,K_trials,n_obs):
    # Lopez de Prado Deflated Sharpe Ratio
    sr=R.mean()/R.std()
    g3=skew(R);g4=kurtosis(R,fisher=False)  # non-excess kurtosis
    # variance of SR estimate (Mertens/Lo)
    var_sr=(1 - g3*sr + (g4-1)/4.0*sr**2)/(n_obs-1)
    sd_sr=np.sqrt(var_sr)
    gamma=0.5772156649
    if K_trials<2:
        sr0=0.0
    else:
        e1=nppf(1-1.0/K_trials); e2=nppf(1-1.0/(K_trials*np.e))
        sr0=sd_sr*((1-gamma)*e1+gamma*e2)
    dsr=ncdf((sr-sr0)/sd_sr)
    return dict(sr_trade=sr,sr0=sr0,sd_sr=sd_sr,DSR=dsr,g3=g3,g4=g4)

print("="*70);print("V3.0 SINGLE-LEG BACKTEST: US100 dip-buy/rip-sell, 200EMA filter")
print("="*70)
res={}
for tf,lab in [("5min","M5"),("15min","M15")]:
    tr=backtest(tf=tf);s=stats(tr,lab);res[lab]=(tr,s)
    print(f"\n--- {lab} ---")
    for k in ["trades","win","avgR","medR","sumR","ret_pct","maxdd_pct","skew","kurt","sr_trade","tpd","sl_pct","tgt_pct","algo_pct","longpct"]:
        print(f"  {k:10s}= {s[k]:.3f}" if isinstance(s[k],float) else f"  {k:10s}= {s[k]}")

print("\n"+"="*70);print("DEFLATED SHARPE RATIO (LdP) — testing overfit to NQ +43% bull run")
print("="*70)
for K in [1,10,50]:
    for lab in ["M5","M15"]:
        tr,s=res[lab]
        d=deflated_sharpe(s["R"],K_trials=max(K,1),n_obs=len(s["R"]))
        print(f"  {lab} K={K:3d}: SR/trade={d['sr_trade']:.4f}  SR0(null max)={d['sr0']:.4f}  DSR={d['DSR']:.3f}")

print("\n"+"="*70);print("LONG-ONLY vs SHORT-ONLY split (is it just long beta?)")
print("="*70)
for lab,tf in [("M5","5min")]:
    tr=backtest(tf=tf)
    for side,nm in [(1,"LONG"),(-1,"SHORT")]:
        sub=tr[tr["sgn"]==side]
        if len(sub): print(f"  {nm}: n={len(sub)} win={100*(sub['R']>0).mean():.1f}% avgR={sub['R'].mean():.3f} sumR={sub['R'].sum():.2f}")

print("\n"+"="*70);print("BERTRAM (2010) OPTIMAL Z-ENTRY WITH COST (single-leg NQ)")
print("="*70)
# dimensionless OU: dY=-Y dτ+√2 dW, steady-state std=1 (z units). enter at -a, exit symmetric +a.
# mu(a) ∝ (m-a-c)/(Erfi(m/√2)-Erfi(a/√2)); optimal symmetric m=-a. cost c in z(std) units.
# estimate cost in std units: c = round-trip NQ bid-ask (pts) / std(spread in pts)
d=merged("5min");w=60
ratio=(d["NQ"]/d["YM"]).rolling(w).mean();spread=d["NQ"]-ratio*d["YM"]
sigma_spread=spread.rolling(w).std().median()  # typical std in NQ pts
nq_bidask=d["NQ_SP"].mean()*POINT
c_bar=nq_bidask/sigma_spread
print(f"  sigma_spread(median, NQ pts)={sigma_spread:.2f}  NQ bid-ask(pts)={nq_bidask:.2f}  -> c_bar={c_bar:.4f} std units")
def neg_mu(a,c):  # a<0 entry, m=-a exit
    m=-a
    num=(m-a-c)
    den=erfi(m/np.sqrt(2))-erfi(a/np.sqrt(2))
    if den<=0: return 1e9
    return -num/den
for c in [0.0,c_bar,2*c_bar,5*c_bar]:
    xb,_=minimize_grid(neg_mu,-4.0,-0.05,args=(c,))
    print(f"  cost c_bar={c:.4f} -> optimal |z_entry|={-xb:.3f}")
