import numpy as np, itertools
from v3bt import backtest, stats, deflated_sharpe
# fair param grid on M15 to test if ANY config survives deflation
zs=[1.5,2.0,2.5]; ats=[2.5,3.5,5.0]; ws=[40,60,90]
configs=list(itertools.product(zs,ats,ws))
K=len(configs)
results=[]
for ze,am,w in configs:
    tr=backtest(tf="15min",z_entry=ze,atr_mult=am,w=w)
    if len(tr)<30: continue
    s=stats(tr,"M15")
    results.append((ze,am,w,s["trades"],s["sumR"],s["ret_pct"],s["maxdd_pct"],s["sr_trade"],s["skew"],s["R"]))
results.sort(key=lambda r:-r[4])
print(f"M15 grid: {K} configs tested")
print(f"{'z':>4}{'atr':>5}{'w':>4}{'trades':>8}{'sumR':>8}{'ret%':>8}{'maxDD%':>8}{'SR/t':>7}{'skew':>7}")
for r in results[:8]:
    print(f"{r[0]:>4.1f}{r[1]:>5.1f}{r[2]:>4d}{r[3]:>8d}{r[4]:>8.1f}{r[5]:>8.1f}{r[6]:>8.1f}{r[7]:>7.3f}{r[8]:>7.2f}")
# DSR on the BEST config, deflated by K = number of configs tried
best=results[0]
d=deflated_sharpe(best[9],K_trials=K,n_obs=len(best[9]))
print(f"\nBEST config (z={best[0]},atr={best[1]},w={best[2]}): SR/trade={d['sr_trade']:.4f}")
print(f"Deflated by K={K} trials -> SR0(null)={d['sr0']:.4f}  DSR={d['DSR']:.3f}")
print(f"(DSR > 0.95 needed to claim real edge; <0.95 = consistent with overfitting)")
