"""Visibility-aware diagnostics: preserve missing epochs and independent arcs."""
import numpy as np
from scipy import stats
from .stationkeeping_diagnostics import acov, holm, stationarity

ANGULAR_STATIONARITY_MIN_SAMPLES = 24


def longest_run(values):
    valid = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False,valid,False]))
    if not len(edges):
        return np.array([])
    runs = list(zip(edges[::2],edges[1::2]))
    a,b = max(runs,key=lambda ab:ab[1]-ab[0])
    return values[a:b]


def stationarity_rows(records):
    """Describe the longest observed run for each angle and identify its arc."""
    rows=[]
    for c in range(records.shape[-1]):
        runs = [longest_run(arc[:, c]) for arc in records]
        arc, segment = max(enumerate(runs), key=lambda pair: len(pair[1]))
        if len(segment) >= ANGULAR_STATIONARITY_MIN_SAMPLES:
            r=stationarity(segment,f'LSO {c//2+1} '+('alpha' if c%2==0 else 'delta'))
            r['contiguous_samples']=len(segment)
            rows.append(r)
        else:
            rows.append(dict(channel=f'LSO {c//2+1} '+('alpha' if c%2==0 else 'delta'),
                             contiguous_samples=len(segment),adf_p=None,kpss_p=None,
                             reading='insufficient contiguous observations'))
        rows[-1]['arc'] = arc
    return rows


def _corr(left,right):
    valid=np.isfinite(left)&np.isfinite(right)
    a,b=left[valid],right[valid]
    if len(a)<8:
        return np.nan,len(a)
    a,b=a-a.mean(),b-b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return (float(a@b/den) if den>0 else np.nan),len(a)


def residual_report(e,commands,lags=12):
    """ACF/CCF at actual lags, without joining across missing observations.

    Ljung-Box uses only the longest contiguous run for each arc/channel, with
    at least 24 samples; it is not applied to a compressed sequence of gaps.
    Holm adjustment covers eligible run tests. A conditional Gaussian Monte
    Carlo maximum-CCF test redraws the independently randomized Gaussian
    commands for a no-association reference while preserving residual serial
    dependence and visibility. The mask is treated as fixed in this screen.
    """
    e=np.asarray(e)[:,1:]  # innovation[k+1] paired with command[k]
    arcs,n,channels=e.shape
    lags=min(lags,n//4)
    acf=np.full((lags,channels),np.nan); pairs=np.zeros_like(acf,dtype=int)
    for lag in range(1,lags+1):
        for c in range(channels):
            acf[lag-1,c],pairs[lag-1,c]=_corr(e[:,lag:,c],e[:,:-lag,c])
    def correlations(inputs):
        result=np.full((lags+1,channels,3),np.nan)
        counts=np.zeros_like(result,dtype=int)
        for lag in range(lags+1):
            for c in range(channels):
                for j in range(3):
                    result[lag,c,j],counts[lag,c,j]=_corr(e[:,lag:,c],inputs[:,:n-lag,j])
        return result,counts
    ccf,ccf_pairs=correlations(commands)
    ccf_p = None
    if np.isfinite(ccf).any():
        observed = np.nanmax(abs(ccf) * np.sqrt(ccf_pairs))
        rng = np.random.default_rng(696012)
        exceed = 0
        repetitions = 199
        for _ in range(repetitions):
            # Pearson correlation is invariant to the command draw's scale.
            null, count = correlations(rng.normal(size=commands.shape))
            exceed += np.nanmax(abs(null) * np.sqrt(count)) >= observed
        ccf_p = (1 + exceed) / (repetitions + 1)
    p=np.full((arcs,channels),np.nan); lengths=np.zeros_like(p,dtype=int)
    for a in range(arcs):
        for c in range(channels):
            run=longest_run(e[a,:,c]); lengths[a,c]=len(run)
            if len(run)<24:
                continue
            m=min(lags,len(run)//4); g=acov(run,m); rho=g[1:]/max(g[0],1e-300)
            Q=len(run)*(len(run)+2)*np.sum(rho**2/(len(run)-np.arange(1,m+1)))
            p[a,c]=stats.chi2.sf(Q,m)
    adjusted=np.full_like(p,np.nan)
    if np.isfinite(p).any():
        adjusted[np.isfinite(p)]=holm(p[np.isfinite(p)])
    return dict(acf=acf,acf_pairs=pairs,ccf=ccf,ccf_pairs=ccf_pairs,
                ljung_box_p=p,ljung_box_holm_p=adjusted,contiguous_lengths=lengths,
                ccf_global_p=ccf_p,
                standardized_mean=np.divide(
                    np.nansum(e, axis=(0, 1)), np.isfinite(e).sum(axis=(0, 1)),
                    out=np.full(channels, np.nan), where=np.isfinite(e).any(axis=(0, 1))),lags=lags,
                eligible_whiteness_tests=int(np.isfinite(p).sum()))
