"""Uncertainty forecasts at exact nominal-period checkpoints."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import numpy as np
from .study import make_truth
from .estimation import run_ekf
from .frames import to_mci, state_transform, acceleration_jacobian


def forecast_horizons(cfg, periods=(1, 5, 10, 20)):
    """Forecast AO/AO+rate covariance continuously to the longest horizon.

    All checkpoints share one prior and all intervening visible observations.
    Period multiples refer to the supplied nominal T, without orbit resets,
    stationkeeping or periodic tiling. This is not an empirical error forecast.
    """
    periods = np.unique(np.asarray(periods, dtype=float))
    if not len(periods) or np.any(~np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError('Horizon periods must be positive finite values')
    config = deepcopy(cfg)
    config['simulation']['periods'] = float(periods[-1])
    checkpoints = periods * config['orbit']['period_tu']
    result = make_truth(config, extra_times=checkpoints)
    result.update(config=config, measurements=None)
    # Do not add extra observations at report times. Forecast to the exact
    # checkpoint, using only samples on the configured cadence beforehand.
    cadence = config['simulation']['cadence_s'] / config['time_unit_s']
    is_scheduled = np.isclose(result['times'] / cadence,
                              np.rint(result['times'] / cadence), rtol=0, atol=1e-9)
    result['visible'] = result['visible'] & is_scheduled
    rows = []
    for name, rates in [('angles', False), ('angles_rates', True)]:
        fit = run_ekf(result['times'], result['truth'], result['visible'],
                      None, config, rates, covariance_only=True)
        result[name] = fit
        for count, epoch in zip(periods, checkpoints):
            k = int(np.searchsorted(result['times'], epoch))
            x, P = fit['state'][k], fit['covariance'][k]
            B = state_transform(epoch, config)
            C = B @ P @ B.T
            G = acceleration_jacobian(epoch, x, config)
            Ca = G @ P @ G.T
            moon_distance = np.linalg.norm(x[:3] - [1-config['mu'], 0, 0]) * config['length_unit_km']
            elapsed = epoch * config['time_unit_s']
            available = np.flatnonzero(result['visible'][:k+1])
            last = float(result['times'][available[-1]] * config['time_unit_s']) if len(available) else None
            sigmas = np.sqrt(np.maximum(np.r_[np.diag(C), np.diag(Ca)], 0))
            row = dict(periods=float(count), days=float(elapsed/86400),
                       measurement_case=name, available_observations=int(len(available)),
                       time_since_last_observation_days=None if last is None else (elapsed-last)/86400,
                       moon_distance_km=float(moon_distance),
                       position_rss_sigma_km=float(np.sqrt(max(np.trace(C[:3,:3]),0))),
                       velocity_rss_sigma_m_s=float(1000*np.sqrt(max(np.trace(C[3:,3:]),0))),
                       acceleration_rss_sigma_m_s2=float(1000*np.sqrt(max(np.trace(Ca),0))))
            for j, axis in enumerate('xyz'):
                row['position_sigma_'+axis+'_km'] = float(sigmas[j])
                row['velocity_sigma_'+axis+'_m_s'] = float(1000*sigmas[j+3])
                row['acceleration_sigma_'+axis+'_m_s2'] = float(1000*sigmas[j+6])
            rows.append(row)
    result['horizon_rows'] = rows
    result['horizon_periods'] = periods
    result['diagnostics']['visible_epochs'] = int(result['visible'].sum())
    return result


def save_horizon_forecast(result, directory):
    """Save summary CSV/JSON and full covariance histories for reproduction."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = result['horizon_rows']
    with (directory/'horizon_uncertainty.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(config=result['config'], diagnostics=result['diagnostics'],
                  interpretation='Local nominal covariance with continuing scheduled LSO measurements; no stationkeeping, orbit reset, or periodic tiling. Not validated estimation errors.',
                  horizon_rows=rows)
    (directory/'horizon_uncertainty.json').write_text(json.dumps(report, indent=2)+'\n')
    arrays = {key: result[key] for key in ['times','truth','visible','elevation']}
    for name in ['angles', 'angles_rates']:
        arrays[name+'_covariance'] = result[name]['covariance']
    np.savez_compressed(directory/'horizon_histories.npz', **arrays)
