#!/usr/bin/env python3
import pandas as pd, numpy as np, os
DATA = "/sessions/festive-funny-gates/mnt/Mft/for claude/DATA FILE"
SPEC = {
 "XAUUSD": dict(f="((now))XAUUSD_M5_202501020840_202606042345.csv", point=0.01,    mult=100.0,    comm_rt=6.0),
 "XAGUSD": dict(f="((now))XAGUSD_M5_202501020640_202606042345.csv", point=0.001,   mult=5000.0,   comm_rt=6.0),
 "EURUSD": dict(f="((now))EURUSD_M5_202501300620_202606050025.csv", point=0.00001, mult=100000.0, comm_rt=6.0),
 "GBPUSD": dict(f="((now))GBPUSD_M5_202501272230_202606050025.csv", point=0.00001, mult=100000.0, comm_rt=6.0),
 "US500":  dict(f="((now))US500.cash_M5_202412310920_202606042345.csv", point=0.01, mult=1.0,     comm_rt=0.0),
 "US100":  dict(f="((now))US100.cash_M5_202412310520_202606042345.csv", point=0.01, mult=1.0,     comm_rt=0.0),
 "US30":   dict(f="US30.cash_M5_202412310445_202606042345.csv",         point=0.01, mult=1.0,     comm_rt=0.0),
}
def load(sym):
    s=SPEC[sym]; df=pd.read_csv(os.path.join(DATA,s["f"]),sep="\t")
    df.columns=[c.strip("<>").upper() for c in df.columns]
    df["TS"]=pd.to_datetime(df["DATE"]+" "+df["TIME"],format="%Y.%m.%d %H:%M:%S")
    return df[["TS","CLOSE","SPREAD"]].rename(columns={"CLOSE":f"P_{sym}","SPREAD":f"SP_{sym}"}).set_index("TS")
def rbeta(y,x,win):
    mx=x.rolling(win).mean();my=y.rolling(win).mean()
    cov=(x*y).rolling(win).mean()-mx*my; var=(x*x).rolling(win).mean()-mx*mx
    return cov/var
def backtest(symA,symB,win=60,z_entry=2.0,z_exit=0.4,z_stop=3.2,notional=25000.0,cost_mult=1.0):
    A,B=load(symA),load(symB); df=A.join(B,how="inner").dropna()
    pA,pB=df[f"P_{symA}"].values,df[f"P_{symB}"].values
    spA,spB=df[f"SP_{symA}"].values,df[f"SP_{symB}"].values
    la,lb=np.log(pA),np.log(pB)
    beta=rbeta(pd.Series(la),pd.Series(lb),win).values
    resid=la-beta*lb; rs=pd.Series(resid)
    z=(resid-rs.rolling(win).mean().values)/rs.rolling(win).std().values
    sA,sB=SPEC[symA],SPEC[symB]
    pos=0;entry_i=entry_z=None;lotsA=lotsB=pA0=pB0=fricA0=fricB0=0.0;sgn=0
    trades=[];n=len(df)
    for i in range(win+1,n):
        if not(np.isfinite(z[i]) and np.isfinite(beta[i])): continue
        if pos==0:
            if abs(z[i])>z_entry and beta[i]>0:
                sgn=-1 if z[i]>0 else 1
                nA=notional;nB=beta[i]*notional
                lotsA=nA/(sA["mult"]*pA[i]);lotsB=nB/(sB["mult"]*pB[i])
                entry_i=i;entry_z=z[i];pA0,pB0=pA[i],pB[i]
                fricA0=lotsA*sA["mult"]*(spA[i]*sA["point"])*0.5+lotsA*sA["comm_rt"]*0.5
                fricB0=lotsB*sB["mult"]*(spB[i]*sB["point"])*0.5+lotsB*sB["comm_rt"]*0.5
                pos=sgn
        else:
            he=abs(z[i])<z_exit;hs=abs(z[i])>z_stop
            if he or hs:
                pnlA=sgn*lotsA*sA["mult"]*(pA[i]-pA0); pnlB=-sgn*lotsB*sB["mult"]*(pB[i]-pB0)
                gross=pnlA+pnlB
                fricA1=lotsA*sA["mult"]*(spA[i]*sA["point"])*0.5+lotsA*sA["comm_rt"]*0.5
                fricB1=lotsB*sB["mult"]*(spB[i]*sB["point"])*0.5+lotsB*sB["comm_rt"]*0.5
                fric=(fricA0+fricB0+fricA1+fricB1)*cost_mult
                trades.append(dict(bars=i-entry_i,entry_z=entry_z,exit_z=z[i],gross=gross,fric=fric,net=gross-fric,stopped=int(hs)))
                pos=0
    return pd.DataFrame(trades),df
def summarize(symA,symB,**kw):
    tr,df=backtest(symA,symB,**kw)
    if len(tr)==0: return dict(pair=f"{symA}/{symB}",trades=0),tr
    g=tr["gross"];f=tr["fric"];net=tr["net"];n=len(tr);half=n//2
    return dict(pair=f"{symA}/{symB}",trades=n,gross_pt=g.mean(),fric_pt=f.mean(),net_pt=net.mean(),
        net_total=net.sum(),winrate=(net>0).mean()*100,ratio=g.mean()/f.mean() if f.mean()>0 else np.nan,
        med_bars=tr["bars"].median(),stop_pct=tr["stopped"].mean()*100,
        net_pt_h1=net.iloc[:half].mean(),net_pt_h2=net.iloc[half:].mean(),
        maxdd=(net.cumsum().cummax()-net.cumsum()).max()),tr
pairs=[("XAUUSD","XAGUSD"),("EURUSD","GBPUSD"),("US100","US30"),("US100","US500"),("US500","US30"),("XAUUSD","US100")]
rows=[]
for a,b in pairs:
    try:
        r,_=summarize(a,b);rows.append(r);print("done",a,b,r["trades"])
    except Exception as e:
        import traceback;print("ERR",a,b,e);traceback.print_exc()
out=pd.DataFrame(rows).sort_values("net_pt",ascending=False)
pd.set_option("display.width",220,"display.max_columns",30)
cols=["pair","trades","gross_pt","fric_pt","net_pt","ratio","winrate","med_bars","stop_pct","net_pt_h1","net_pt_h2","net_total","maxdd"]
print();print(out[cols].to_string(index=False,float_format=lambda x:f"{x:,.2f}"))
out.to_csv("/sessions/festive-funny-gates/mnt/outputs/screen_results.csv",index=False)
