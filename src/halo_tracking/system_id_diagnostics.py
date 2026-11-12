"""Studio-inspired diagnostics for independent, phase-aligned guidance arcs."""

import numpy as np
from scipy import signal

from .frames import state_transform
from .identification import _forward


def lagged(values, steps, delay=0):
    """Newest-first regressors, with unavailable rows padded inside each arc."""
    history = np.full((*values.shape[:2], steps, values.shape[-1]), np.nan)
    for tap in range(min(steps, max(0, values.shape[1] - delay))):
        lag = tap + delay
        history[:, lag:, tap] = values[:, :values.shape[1] - lag]
    return history.reshape(*values.shape[:2], -1)


def design_spectrum(design):
    """Studio 04 conditioning check on centered, unit-RMS columns.

    Keep constant columns as zero columns so missing excitation costs rank.
    The tolerance is a screening convention, relative to the leading value.
    """
    matrix = design[np.isfinite(design).all(axis=-1)]
    if not len(matrix):
        return np.array([]), 0, design.shape[-1], 0
    centered = matrix - matrix.mean(axis=0)
    scale = np.sqrt(np.mean(centered**2, axis=0))
    active = scale > 1e-12 * np.max(np.abs(matrix), axis=0)
    normalized = np.divide(centered, scale, out=np.zeros_like(centered), where=active)
    values = np.linalg.svd(normalized, compute_uv=False)
    values = np.pad(values, (0, matrix.shape[-1] - len(values)))
    relative = values / values[0] if values[0] else values
    return relative, int(np.count_nonzero(relative > 1e-3)), matrix.shape[-1], len(matrix)


def _welch_windows(values, length):
    windows = np.lib.stride_tricks.sliding_window_view(values, length, axis=1)
    return windows[:, ::max(1, length // 2)]


def ensemble_welch(values, fs, length):
    """Studio 08 Welch averaging, adapted to independent arcs and gaps.

    Use Hann windows and 50% overlap. SciPy supplies window-power normalization
    and the one-sided density convention. Each window has its own mean removed.
    """
    windows = _welch_windows(values, length)
    usable = np.isfinite(windows).all(axis=(-1, -2))
    segments = windows[usable]
    if not len(segments):
        return np.fft.rfftfreq(length, 1 / fs), np.full((values.shape[-1], length // 2 + 1), np.nan), 0, 0
    f, densities = signal.periodogram(segments, fs=fs, window="hann", axis=-1)
    return f, densities.mean(axis=0), len(segments), int(usable.any(axis=1).sum())


def ensemble_frequency_response(x, y, fs, length):
    """Studio 08 PSDs, H1 and coherence on matched within-arc Welch windows.

    Average auto- and cross-spectra before forming H1 and coherence. A
    single-window coherence estimate is uninformative.
    """
    x_windows, y_windows = _welch_windows(x, length), _welch_windows(y, length)
    usable = (np.isfinite(x_windows).all(axis=(-1, -2))
              & np.isfinite(y_windows).all(axis=(-1, -2)))
    count, arcs = int(usable.sum()), int(usable.any(axis=1).sum())
    if count < 2:
        shape = (x.shape[-1], length // 2 + 1)
        missing = np.full(shape, np.nan)
        return np.fft.rfftfreq(length, 1 / fs), missing, missing, missing, missing, count, arcs
    left, right = x_windows[usable], y_windows[usable]
    f, xx = signal.periodogram(left, fs=fs, window="hann", axis=-1)
    _, yy = signal.periodogram(right, fs=fs, window="hann", axis=-1)
    _, xy = signal.csd(left, right, fs=fs, window="hann", nperseg=length, axis=-1)
    suu, syy, suy = xx.mean(axis=0), yy.mean(axis=0), xy.mean(axis=0)
    h1 = np.divide(suy, suu, out=np.full_like(suy, np.nan), where=suu > 0)
    denominator = suu * syy
    coherence = np.divide(np.abs(suy)**2, denominator,
                          out=np.full_like(denominator, np.nan), where=denominator > 0)
    return f, suu, syy, h1, np.clip(coherence, 0, 1), count, arcs


def ensemble_coherence(x, y, fs, length):
    """Return the coherence and support from the matched Welch estimate."""
    f, _, _, _, coherence, count, arcs = ensemble_frequency_response(x, y, fs, length)
    return f, coherence, count, arcs


def lag_correlation(response, regressor, lags):
    """Studio 08 CCF / Studios 10 and 12 residual-input check for gapped arcs.

    Compute Pearson correlations on finite within-arc pairs e[k], x[k-lag].
    Report sample and episode support. Positive lags use earlier regressors.
    """
    corr = np.full((len(lags), response.shape[-1], regressor.shape[-1]), np.nan)
    pairs = np.zeros(len(lags), dtype=int)
    arcs = pairs.copy()
    for i, lag in enumerate(lags):
        a = response[:, lag:] if lag else response
        b = regressor[:, :-lag] if lag else regressor
        keep = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
        pairs[i] = keep.sum()
        arcs[i] = keep.any(axis=1).sum()
        if pairs[i] < 3:
            continue
        left, right = a[keep], b[keep]
        left = left - left.mean(axis=0)
        right = right - right.mean(axis=0)
        scale = np.outer(np.linalg.norm(left, axis=0), np.linalg.norm(right, axis=0))
        np.divide(left.T @ right, scale, out=corr[i], where=scale > 0)
    return corr, pairs, arcs


def residual_checks(residual, nav, burns, repetitions=399):
    """Held-out residual diagnostics with an epoch-conditioned permutation check.

    Shuffle entire residual vectors among eligible episodes at each epoch.
    The null requires exchangeability across those episodes and independence
    across epochs (ACF) or from the chosen regressor (CCF). Keeping the masks
    and epoch fixed preserves phase and the number of available pairs.
    """
    lags = np.arange(1, min(4, residual.shape[1] - 1) + 1)
    valid = np.isfinite(residual).all(axis=-1)
    epoch_rows = [np.flatnonzero(column) for column in valid.T]
    can_shuffle = sum(len(rows) > 1 for rows in epoch_rows) >= 2
    rng = np.random.default_rng(1022)
    permutations = []
    if can_shuffle:
        for _ in range(repetitions):
            shuffled = residual.copy()
            for k, rows in enumerate(epoch_rows):
                shuffled[rows, k] = residual[rng.permutation(rows), k]
            permutations.append(shuffled)

    def summarize(regressor, selected_lags):
        def correlations(errors):
            values, pairs, arcs = lag_correlation(
                errors, errors if regressor is None else regressor, selected_lags)
            if regressor is None:
                values = np.diagonal(values, axis1=1, axis2=2)
            return values, pairs, arcs

        values, pairs, arcs = correlations(residual)
        supported = (pairs >= 8) & (arcs >= 3)
        checked = values[supported]
        if not can_shuffle or not np.isfinite(checked).any():
            return selected_lags, values, pairs, arcs, supported, None
        observed = np.nanmax(np.abs(checked))
        exceed = 0
        for shuffled in permutations:
            candidate = correlations(shuffled)[0][supported]
            if np.isfinite(candidate).any():
                exceed += np.nanmax(np.abs(candidate)) >= observed
        return selected_lags, values, pairs, arcs, supported, (1 + exceed) / (repetitions + 1)

    return {
        "Residual autocorrelation": summarize(None, lags),
        "Residual vs. current and past navigation": summarize(nav, np.r_[0, lags]),
        "Residual vs. prior executed burns": summarize(burns, lags),
    }


def nominal_stm_correction(data, name, cfg, period):
    """Fixed-horizon, immediate-burn proxy computed from the known reference.

    The STM is integrated about the saved nominal state at each decision time.
    A three-component regularized correction compresses its action on the
    current EKF deviation. No teacher timing, feasibility, or truth is used.
    """
    settings = cfg["narx"]
    guidance = settings[name + "_guidance"]
    dt = settings["guidance_horizon_fraction"] * period
    ref = data[name + "_reference_normalized"]
    deviation = data[name + "_navigation_normalized"] - ref
    weights = np.repeat([1 / guidance["position_tolerance_km"],
                         1000 / guidance["velocity_tolerance_m_s"]], 3)
    velocity_scale = cfg["time_unit_s"] / (1000 * cfg["length_unit_km"])
    reg = settings["regularization"] / guidance["maximum_impulse_m_s"]
    gains = []
    for t, state in zip(data["decision_times_tu"], ref[0]):
        _, phi = _forward(state, t, dt, cfg, True)
        terminal = weights[:, None] * (state_transform(t + dt, cfg) @ phi)
        impulse = terminal[:, 3:] * velocity_scale
        gain = np.linalg.lstsq(
            np.vstack((impulse, reg * np.eye(3))),
            np.vstack((-terminal, np.zeros((3, 6)))), rcond=None,
        )[0]
        gains.append(gain)
    return np.einsum("tij,atj->ati", gains, deviation)


def guidance_designs(data, name, cfg, period):
    """Nested phase and history comparisons on deployable guidance inputs."""
    nav = data[name + "_navigation_error_km_m_s"]
    burns = data[name + "_applied_delta_v_rotating_m_s"]
    steps = int(cfg["narx"]["history_steps"])
    phase = data["decision_phase_rad"]
    phase = np.broadcast_to(np.column_stack((np.sin(phase), np.cos(phase))), (*nav.shape[:2], 2))
    phase_interactions = (nav[..., :, None] * phase[..., None, :]).reshape(*nav.shape[:2], -1)
    current = np.concatenate((nav, phase, phase_interactions), axis=-1)
    proxy = nominal_stm_correction(data, name, cfg, period)
    designs = {"Current navigation": nav, "Phase-aware current": current}
    for order in sorted({min(2, steps), steps} - {1}):
        designs[f"Phase + {order} navigation taps"] = np.concatenate(
            (current, lagged(nav, order - 1, delay=1)), axis=-1)
    history = designs.get(f"Phase + {steps} navigation taps", current)
    designs["Navigation + prior executed burns"] = np.concatenate(
        (history, lagged(burns, steps, delay=1)), axis=-1)
    designs["Phase + current STM proxy"] = np.concatenate((current, proxy), axis=-1)
    designs["Navigation history + current STM proxy"] = np.concatenate((history, proxy), axis=-1)
    if steps > 1:
        designs["Navigation history + STM proxy history"] = np.concatenate(
            (history, lagged(proxy, steps)), axis=-1)
    return designs, proxy


def feasible_targets(data, name):
    """Feasible triggered labels for burn-vector regression on common rows."""
    burn = data[name + "_target_delta_v_rotating_m_s"]
    mask = (data["history_valid"][None, :] & data[name + "_maneuver_required"]
            & data[name + "_maneuver_valid"] & ~data[name + "_scheduled_pending"]
            & np.isfinite(burn).all(axis=-1))
    return burn, mask


def fit_ridge(design, target, penalty=1.0):
    """Studio 04 least squares with ridge augmentation and training scaling."""
    mean, scale = design.mean(axis=0), design.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    x = (design - mean) / scale
    offset = target.mean(axis=0)
    coefficients = np.linalg.lstsq(
        np.vstack((x, np.sqrt(penalty) * np.eye(x.shape[1]))),
        np.vstack((target - offset, np.zeros((x.shape[1], target.shape[1])))),
        rcond=None,
    )[0]

    def predict(values):
        return (values - mean) / scale @ coefficients + offset

    return predict
