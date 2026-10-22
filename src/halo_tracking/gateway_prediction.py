"""Blind-prediction sensitivity study with unmodeled Gateway acceleration.

The observer continues its uncontrolled trajectory. Gateway truth receives
a configurable small inertial acceleration, but both EKFs use unforced CR3BP.
A forced Gateway measurement blackout isolates predictive endurance without
resetting either filter or its covariance.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .frames import state_transform
from .gateway_comparison import run_gateway_comparison, comparison_metrics
from .orbit_reference import correct_halo_reference

METHOD = 'gateway_forced_blackout_mismatched_truth_v1'
CASES = ('perfect', 'lso_estimated')


def _signature(cfg):
    return hashlib.sha256(json.dumps(
        cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def time_since_last_observation(times, availability, time_unit_s):
    """Days since the last accepted sample; NaN until first observation."""
    times = np.asarray(times, dtype=float)
    availability = np.asarray(availability, dtype=bool)
    if times.ndim != 1 or availability.shape != times.shape:
        raise ValueError('Times and visibility must be equal-sized vectors')
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Times must be finite and strictly increasing')
    days = np.full(len(times), np.nan)
    last = None
    for i, t in enumerate(times):
        if availability[i]:
            last = t
        if last is not None:
            days[i] = (t-last)*time_unit_s/86400.
    return days


def rolling_rmse(times, position_error, window_tu):
    """Time-window RMS of actual vector-error magnitude, including endpoint."""
    times = np.asarray(times, dtype=float)
    error = np.asarray(position_error, dtype=float)
    if (times.shape != error.shape or times.ndim != 1 or window_tu <= 0
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid rolling time series or window')
    squared = error**2
    sums = np.r_[0., np.cumsum(squared)]
    left = np.searchsorted(times, times-window_tu, side='left')
    count = np.arange(1, len(times)+1) - left
    return np.sqrt((sums[1:] - sums[left])/count)


def run_gateway_prediction_study(cfg, periods=20, blackout_period=8,
                                 disturbance_m_s2=2e-9,
                                 rolling_window_days=1.,
                                 thresholds_km=(1., 10., 100.)):
    """Run one physically perturbed-truth study, report how blind error grows.

    Blackout is an explicitly imposed scenario from observer time blackout*T
    onwards, regardless of geometric visibility. It is NOT inferred from the
    actual observational schedule. LSO measurements remain governed by real
    geometric masks and range limits.
    """
    if (periods < 2 or int(periods) != periods or blackout_period < 1
            or int(blackout_period) != blackout_period or blackout_period >= periods):
        raise ValueError('Need at least 2 periods and 1 <= blackout_period < periods')
    if not np.isfinite(disturbance_m_s2) or disturbance_m_s2 <= 0:
        raise ValueError('Disturbance must be strictly positive for mismatch study')
    if not np.isfinite(rolling_window_days) or rolling_window_days <= 0:
        raise ValueError('Rolling window must be positive')
    limits = tuple(float(x) for x in thresholds_km)
    if not limits or not all(np.isfinite(x) and x > 0 for x in limits):
        raise ValueError('Positive finite position-error thresholds required')
    nominal_period = correct_halo_reference(cfg)['orbit']['period_tu']
    cutoff = blackout_period*nominal_period
    # Observer's truth is still the original rounded catalog seed.
    tracking_cfg = json.loads(json.dumps(cfg))
    tracking_cfg['orbit']['period_tu'] = nominal_period
    result = run_gateway_comparison(
        tracking_cfg, duration_periods=int(periods),
        gateway_truth_acceleration_m_s2=float(disturbance_m_s2),
        gateway_measurement_cutoff_tu=cutoff)
    times = np.asarray(result['times'])
    availability = np.asarray(result['gateway_available'], dtype=bool)
    age = time_since_last_observation(
        times, availability, cfg['time_unit_s'])
    day = times*cfg['time_unit_s']/86400.
    last = np.flatnonzero(availability & (times <= cutoff+1e-10))
    last_index = int(last[-1]) if len(last) else None
    if np.any(availability & (times > cutoff+1e-10)):
        raise RuntimeError('Gateway measurement admitted after blackout')
    metrics = comparison_metrics(result)
    signals = dict(
        times_tu=times, elapsed_days=day,
        gateway_available=availability.astype(np.uint8),
        lso_available=np.asarray(result['lso_available'], dtype=np.uint8),
        time_since_gateway_observation_days=age)
    window_tu = rolling_window_days*86400/cfg['time_unit_s']
    for name in CASES:
        data = metrics[name]
        position = data['position_error_km']
        signals[name+'_position_error_km'] = position
        signals[name+'_velocity_error_m_s'] = data['velocity_error_m_s']
        signals[name+'_rolling_position_rmse_km'] = rolling_rmse(
            times, position, window_tu)
        # Use covariance propagated in the actual EKF, not the recentered
        # nominal forecast. This remains a conditional linearized model.
        sigma = []
        for t, P in zip(times, result[name]['covariance']):
            B = state_transform(t, result['config'])
            C = B@P@B.T
            sigma.append(np.sqrt(max(np.trace(C[:3, :3]), 0.)))
        signals[name+'_filter_position_sigma_km'] = np.asarray(sigma)

    rows = []
    for k in range(1, int(periods)+1):
        endpoint = k*nominal_period
        idx = int(np.argmin(np.abs(times-endpoint)))
        if abs(times[idx]-endpoint) > 1e-8:
            raise RuntimeError(f'Missing exact horizon {k}T')
        for name in CASES:
            rows.append(dict(
                horizon_periods=k, case=name,
                gateway_observations_total=int(availability[:idx+1].sum()),
                gateway_observations_latest_period=int(np.sum(
                    availability & (times > (k-1)*nominal_period+1e-10) &
                    (times <= endpoint+1e-10))),
                days_since_gateway_observation=float(age[idx]),
                instantaneous_position_error_km=float(
                    signals[name+'_position_error_km'][idx]),
                rolling_position_rmse_km=float(
                    signals[name+'_rolling_position_rmse_km'][idx]),
                position_sigma_km=float(
                    signals[name+'_filter_position_sigma_km'][idx])))

    threshold_rows = []
    for name in CASES:
        error = signals[name+'_position_error_km']
        for limit in limits:
            exceeded = np.flatnonzero((times >= cutoff-1e-10) &
                                      (error >= limit))
            j = int(exceeded[0]) if len(exceeded) else None
            if j is None:
                status, crossing, since = 'not_reached', None, None
            else:
                status, crossing = 'reached', float(day[j])
                since = (float(age[j]) if np.isfinite(age[j]) else None)
            threshold_rows.append(dict(
                case=name, position_threshold_km=limit, status=status,
                crossing_elapsed_days=crossing,
                crossing_days_since_last_observation=since))

    summary = dict(
        method=METHOD, config_signature=_signature(cfg),
        periods=int(periods), blackout_period=int(blackout_period),
        blackout_elapsed_days=float(cutoff*cfg['time_unit_s']/86400),
        last_accepted_gateway_observation_days=(
            float(day[last_index]) if last_index is not None else None),
        disturbance_m_s2=float(disturbance_m_s2),
        disturbance_direction_mci=cfg.get('gateway_truth_disturbance', {}).get(
            'direction_mci', [1., .3, -.2]),
        rolling_window_days=float(rolling_window_days),
        thresholds_km=list(limits),
        nominal_period_tu=float(nominal_period),
        description=('Illustrative deterministic constant inertial acceleration '
                     'is applied to Gateway TRUTH only. The EKF uses CR3BP and '
                     'no Gateway process noise. Gateway optical measurements '
                     'are forcibly disabled after the selected period. '
                     'LSO geometry and Sun/Earth keep-outs remain active. '
                     'Error is from one noise realization; sigma is conditional '
                     'linearized filter covariance and cannot represent truth '
                     'model error. Non-crossings are right-censored by study end.'))
    return dict(summary=summary, histories=signals, rows=rows,
                threshold_rows=threshold_rows)


def save_gateway_prediction_study(study, directory):
    """Save exact signals and both CSV summaries for reproducible inspection."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'gateway_prediction_histories.npz',
                        **study['histories'])
    for filename, rows in [
        ('gateway_prediction_horizons.csv', study['rows']),
        ('gateway_prediction_thresholds.csv', study['threshold_rows'])]:
        with (directory/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory/'gateway_prediction_summary.json').write_text(
        json.dumps(study['summary'], indent=2)+'\n')


def load_gateway_prediction_study(cfg, directory, periods=20,
                                  blackout_period=8, disturbance_m_s2=2e-9,
                                  rolling_window_days=1.,
                                  thresholds_km=(1., 10., 100.)):
    """Reuse a completed sensitivity run only with fully matching parameters."""
    directory = Path(directory)
    summary_path = directory/'gateway_prediction_summary.json'
    files = [summary_path, directory/'gateway_prediction_histories.npz',
             directory/'gateway_prediction_horizons.csv',
             directory/'gateway_prediction_thresholds.csv']
    if not all(path.is_file() for path in files):
        return None
    try:
        summary = json.loads(summary_path.read_text())
        if (summary.get('method') != METHOD
                or summary.get('config_signature') != _signature(cfg)
                or summary.get('periods') != int(periods)
                or summary.get('blackout_period') != int(blackout_period)
                or summary.get('disturbance_m_s2') != float(disturbance_m_s2)
                or summary.get('rolling_window_days') != float(rolling_window_days)
                or summary.get('thresholds_km') != list(map(float, thresholds_km))):
            return None
        def read_csv(filename):
            with (directory/filename).open(newline='') as stream:
                return list(csv.DictReader(stream))
        rows = read_csv('gateway_prediction_horizons.csv')
        thresholds = read_csv('gateway_prediction_thresholds.csv')
        if len(rows) != 2*int(periods) or len(thresholds) != 2*len(thresholds_km):
            return None
        for i, row in enumerate(rows):
            if (int(row['horizon_periods']) != i//2+1
                    or row['case'] != CASES[i%2]):
                return None
            row['horizon_periods'] = int(row['horizon_periods'])
            row['gateway_observations_total'] = int(row['gateway_observations_total'])
            row['gateway_observations_latest_period'] = int(
                row['gateway_observations_latest_period'])
            for key in ('days_since_gateway_observation',
                        'instantaneous_position_error_km',
                        'rolling_position_rmse_km', 'position_sigma_km'):
                row[key] = float(row[key])
        for row in thresholds:
            row['position_threshold_km'] = float(row['position_threshold_km'])
            for key in ('crossing_elapsed_days',
                        'crossing_days_since_last_observation'):
                row[key] = None if row[key] in ('', 'None') else float(row[key])
        with np.load(directory/'gateway_prediction_histories.npz',
                     allow_pickle=False) as archive:
            histories = {name: archive[name].copy() for name in archive.files}
        if len(histories['times_tu']) < 2:
            return None
        return dict(summary=summary, histories=histories, rows=rows,
                    threshold_rows=thresholds)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
