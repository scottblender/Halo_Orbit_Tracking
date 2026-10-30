"""Matched nonlinear orbit test conditional on the STM reference burn schedule."""
import json
from pathlib import Path
import numpy as np
from .year_study import _observer_reference, _initial_conditions, simulate_one_arc
from .gateway_reference_periodic import gateway_periodic_reference
from .network import sensor_configs


def run_polynomial_pair(cfg,bundle,output_dir,days=365.,seed=24017):
    if bundle['configuration'] != cfg['narx']:
        raise ValueError('Training/deployment configuration mismatch')
    if days<=0:
        raise ValueError('days must be positive')
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    observer,T,corrected = _observer_reference(cfg)
    gateway,_ = gateway_periodic_reference(cfg)
    end = days*86400/cfg['time_unit_s']
    grid = np.r_[np.arange(0.,end,T/cfg['narx']['decisions_per_period']),end]
    cadence = cfg['narx']['closed_loop_measurement_cadence_s']/cfg['time_unit_s']
    observations = np.unique(np.r_[np.arange(0.,end,cadence),grid])
    initial = _initial_conditions(cfg,np.asarray(corrected['orbit']['state']),gateway(0),
        seed,len(observations),len(sensor_configs(cfg)))
    np.savez_compressed(out/'paired_conditions.npz',truth=initial[0],estimate=initial[1],
        covariance=initial[2],optical_noise=initial[3],camera_noise=initial[4])
    episodes = {}; summary = {}
    for label,controller in [('STM','stm'),('Polynomial NARX','polynomial_narx')]:
        deployment = dict(bundle=bundle,schedule=episodes.get('STM')) if controller=='polynomial_narx' else None
        e = simulate_one_arc(cfg,controller,observer,T,gateway,grid,observations,initial,model=deployment)
        episodes[label] = e
        np.savez_compressed(out/('stm_orbit.npz' if label=='STM' else 'polynomial_orbit.npz'),
            **{k:v for k,v in e.items() if isinstance(v,np.ndarray)})
        p = e['observer_truth_position_departure_km']; v = e['observer_truth_velocity_departure_m_s']
        finite = np.isfinite(p) & np.isfinite(v)
        if not finite.any():
            raise RuntimeError(label+' produced no evaluated states')
        summary[label] = dict(completed_requested_span=e['completed_full_span'],
            last_simulated_day=float(e['day'][finite][-1]),stop_reason=e['halted_reason'] or 'completed requested span',
            maximum_position_departure_km=float(p[finite].max()),maximum_velocity_departure_m_s=float(v[finite].max()),
            cumulative_delta_v_m_s=float(e['delta_v_cost_m_s'].sum()),
            rejected_burns=int(e['maneuver_rejected'].sum()),executed_burns=int(e['executed_burn_count'].sum()))
        if controller=='polynomial_narx':
            z = e['polynomial_input_max_abs_z']; finite_z = np.isfinite(z)
            summary[label]['maximum_input_training_standard_deviations'] = float(z[finite_z].max()) if finite_z.any() else None
            summary[label]['first_day_above_10_training_standard_deviations'] = float(e['day'][finite_z & (z>10)][0]) if np.any(finite_z & (z>10)) else None
        print(label+': '+json.dumps(summary[label]),flush=True)
        if label=='STM' and not e['completed_full_span']:
            raise RuntimeError('STM schedule incomplete; cannot evaluate the full supplied schedule')
    summary['interpretation'] = 'Conditional delta-v test with supplied STM timing; NOT autonomous NARX station keeping. Common STM startup for first four decisions.'
    (out/'orbit_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    return episodes,summary
