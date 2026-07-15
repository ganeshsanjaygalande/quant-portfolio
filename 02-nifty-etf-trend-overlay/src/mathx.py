import numpy as np, math
def erf(x): return math.erf(x)
def ncdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def nppf(p):
    # Acklam inverse normal CDF
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if p<pl:
        q=math.sqrt(-2*math.log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p<=1-pl:
        q=p-0.5;r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q=math.sqrt(-2*math.log(1-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
def erfi(x):
    # (2/sqrt(pi)) * integral_0^x e^{t^2} dt , numeric trapezoid
    if x==0: return 0.0
    s=1 if x>0 else -1; x=abs(x)
    t=np.linspace(0,x,2000); y=np.exp(t*t)
    val=np.trapz(y,t)
    return s*2/math.sqrt(math.pi)*val
def skew(a):
    a=np.asarray(a,float);m=a.mean();s=a.std()
    return ((a-m)**3).mean()/s**3 if s>0 else 0.0
def kurtosis(a,fisher=True):
    a=np.asarray(a,float);m=a.mean();s=a.std()
    k=((a-m)**4).mean()/s**4 if s>0 else 0.0
    return k-3 if fisher else k
def minimize_grid(f,lo,hi,args=(),n=400):
    xs=np.linspace(lo,hi,n);best=None
    for x in xs:
        v=f(x,*args)
        if best is None or v<best[1]: best=(x,v)
    # refine
    x0=best[0];step=(hi-lo)/n
    xs=np.linspace(x0-step,x0+step,n)
    for x in xs:
        v=f(x,*args)
        if v<best[1]: best=(x,v)
    return best
