"""Course diagnostics adapted to vector signals and separate orbit experiments.

Adapted from supplied Studios 06 (acov, segment_stats, unit_root_tests), 08
(welch_est, spectrum_report), and 10/12 (whiteness, xcorr_test, Ljung-Box).
Student TODO functions are completed here; they are not represented as copied
solutions. Spectra use SciPy one-sided density, including window power.
"""
import warnings
import numpy as np
from scipy import signal, stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def acov(x, max_lag):
    """Studio 06: biased 1/N autocovariance, mean removed."""
    x = np.asarray(x, float)-np.mean(x)
    n = x.size
    return signal.correlate(x, x, mode='full', method='fft')[n-1:n+max_lag]/n


def segment_stats(y, n_seg=4):
    """Studio 06 Bartlett-taper correlation-aware mean/variance bands."""
    L = len(y)//n_seg
    if L < 4:
        raise ValueError('Too few samples for segment diagnostics')
    M = min(L//4, len(y)-1)
    segments = np.asarray(y[:L*n_seg]).reshape(n_seg, L)
    g = acov(y[:L*n_seg], M)
    w = 1-np.arange(M+1)/(M+1)
    lrv = max(g[0]+2*np.sum(w[1:]*g[1:]), 1e-300)
    s4 = g[0]**2+2*np.sum(w[1:]*g[1:]**2)
    return dict(mu=segments.mean(axis=1), var=segments.var(axis=1, ddof=1),
                se_mu=np.sqrt(lrv/L), se_var=np.sqrt(2*s4/L),
                G=np.asarray([acov(z, M) for z in segments]), L=L, M=M)


def stationarity(y, name):
    """Opposite nulls: ADF unit root, KPSS level-stationary; neither proves WSS."""
    st = segment_stats(y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a = adfuller(y, regression='c', autolag='AIC')
        k = kpss(y, regression='c', nlags='auto')
    joint = {(True, False): 'supports level stationarity',
             (True, True): 'conflicting tests',
             (False, True): 'evidence against level stationarity',
             (False, False): 'inconclusive / low power'}[(a[1]<.05, k[1]<.05)]
    return dict(channel=name, adf_p=float(a[1]), kpss_p=float(k[1]),
                kpss_warning='; '.join(str(w.message) for w in caught if issubclass(w.category, InterpolationWarning)), reading=joint,
                mean_flags=int(np.sum(np.abs(st['mu']-st['mu'].mean()) > 1.96*st['se_mu'])),
                variance_flags=int(np.sum(np.abs(st['var']-st['var'].mean()) > 1.96*st['se_var'])))


def welch_bundle(u, y, fs, length=32, overlap=.5):
    """Welch on within-arc windows; S_yu=Y conj(U), H=S_yu S_uu^-1.

    u and y are aligned PRE-burn signals of the same length. Full spectral
    inversion keeps three simultaneous inputs separate. Diagonal pairwise
    coherence is a screen, not a test that the MIMO model is correct.
    """
    if u.ndim != 3 or y.ndim != 3 or u.shape[:2] != y.shape[:2]:
        raise ValueError('Use matching [arc,time,channel] arrays')
    if not np.isfinite(u).all() or not np.isfinite(y).all():
        raise ValueError('Gaps require explicit experiment segmentation')
    if length < 4 or length > u.shape[1] or not 0 <= overlap < 1:
        raise ValueError('Invalid Welch length or overlap')
    stride = length-int(round(overlap*length))
    if stride < 1:
        raise ValueError('Overlap leaves no step')
    window = signal.get_window('hann', length)
    def fft_windows(a):
        pieces = np.stack([a[:, j:j+length] for j in range(0, a.shape[1]-length+1, stride)], axis=1)
        pieces = pieces.reshape(-1, length, a.shape[-1])
        pieces -= pieces.mean(axis=1, keepdims=True)
        return np.fft.rfft(window[None, :, None]*pieces, axis=1)
    U, Y = fft_windows(u), fft_windows(y)
    count = U.shape[0]
    k0 = u.shape[0]*(u.shape[1]//length)
    factor = np.ones(length//2+1)*2/(fs*np.sum(window**2))
    factor[0] /= 2
    if length % 2 == 0:
        factor[-1] /= 2
    suu = np.einsum('sfi,sfj->fij', U, U.conj())/count*factor[:, None, None]
    syu = np.einsum('sfi,sfj->fij', Y, U.conj())/count*factor[:, None, None]
    syy = np.einsum('sfi,sfj->fij', Y, Y.conj())/count*factor[:, None, None]
    su = np.diagonal(suu, axis1=1, axis2=2).real
    sy = np.diagonal(syy, axis1=1, axis2=2).real
    cond = np.linalg.cond(suu)
    H = syu @ np.linalg.pinv(suu, rcond=1e-8)
    supported = (cond < 1e6) & (count >= 2*u.shape[-1])
    H[~supported] = np.nan
    denom = sy[:, :, None]*su[:, None, :]
    coh = np.clip(np.divide(abs(syu)**2, denom, out=np.zeros_like(denom), where=denom>0), 0, 1)
    nu = 2*k0  # Studio 08 conservative non-overlap count; nominal WSS interpretation
    band = np.array([nu/stats.chi2.ppf(.975, nu), nu/stats.chi2.ppf(.025, nu)])
    return dict(f=np.fft.rfftfreq(length, 1/fs), suu=su, syy=sy, H=H,
                coherence=coh, input_spectral_condition=cond, supported=supported,
                segments=count, nonoverlap_segments=k0, dof=nu,
                psd_band_factor=band, length=length, overlap=overlap)


def holm(p):
    """Holm adjusted p-values, for a declared family of diagnostic tests."""
    p = np.asarray(p, float)
    flat = p.ravel()
    order = np.argsort(flat)
    adjusted = np.minimum(1, np.maximum.accumulate((len(flat)-np.arange(len(flat)))*flat[order]))
    out = np.empty_like(flat)
    out[order] = adjusted
    return out.reshape(p.shape)


def residual_report(innovations, commands, lags=12):
    """Held-out, covariance-standardized innovations; no subtraction of fit df.

    Ljung-Box is computed separately per arc/channel (fixed trained model).
    CCF pools only valid WITHIN-arc pairs e[k+1],u[k-lag]. p-values are nominal
    large-sample screens; multiple testing uses Holm within each family.
    """
    arcs, n, channels = innovations.shape
    lags = min(lags, n//4)
    rho = np.empty((arcs, channels, lags))
    p = np.empty((arcs, channels))
    for a in range(arcs):
        for c in range(channels):
            g = acov(innovations[a, :, c], lags)
            r = g[1:]/max(g[0], 1e-300)
            rho[a, c] = r
            Q = n*(n+2)*np.sum(r*r/(n-np.arange(1, lags+1)))
            p[a, c] = stats.chi2.sf(Q, lags)
    ccf = np.empty((lags+1, channels, commands.shape[-1]))
    pairs = []
    for lag in range(lags+1):
        left = innovations[:, lag:]
        right = commands[:, :n-lag]
        left = (left-left.mean(axis=1, keepdims=True)).reshape(-1, channels)
        right = (right-right.mean(axis=1, keepdims=True)).reshape(-1, commands.shape[-1])
        denominator = np.outer(np.linalg.norm(left, axis=0), np.linalg.norm(right, axis=0))
        ccf[lag] = np.divide(left.T @ right, denominator,
                             out=np.zeros_like(denominator), where=denominator>0)
        pairs.append(len(left))
    pccf = 2*stats.norm.sf(abs(ccf)*np.sqrt(np.asarray(pairs))[:, None, None])
    return dict(acf=rho, ljung_box_p=p, ljung_box_holm_p=holm(p),
                ccf=ccf, ccf_pairs=np.asarray(pairs), ccf_holm_p=holm(pccf),
                standardized_mean=innovations.mean(axis=(0, 1)),
                standardized_covariance=np.cov(innovations.reshape(-1, channels).T),
                lags=lags, alpha=.05,
                interpretation='Noise-model diagnostics; not proof of dynamics or corridor success')


def lag_response(u, y, max_lag=16):
    """Exploratory CCF, no joining across arcs; positive lag means past command."""
    result = []
    for lag in range(max_lag+1):
        left = y[:, lag:]-y[:, lag:].mean(axis=1, keepdims=True)
        right = u[:, :u.shape[1]-lag]-u[:, :u.shape[1]-lag].mean(axis=1, keepdims=True)
        a, b = left.reshape(-1, y.shape[-1]), right.reshape(-1, u.shape[-1])
        d = np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
        result.append(np.divide(a.T @ b, d, out=np.zeros_like(d), where=d>0))
    return np.asarray(result)
