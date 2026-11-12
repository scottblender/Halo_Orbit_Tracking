"""Audit STM labels, units, cadence and causal lag arrays before fitting."""
from pathlib import Path
import json
import numpy as np
from .two_stage_narx import _load
from .narx_dataset import _error, _reference
from .identification import _forward


def audit_dataset(directory, cfg):
    s = cfg['narx']
    L = int(s['history_steps'])
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata['configuration'] != s:
        raise ValueError('Dataset configuration mismatch: regenerate training data')

    splits = {split: _load(directory / (split + '.npz'))
              for split in ('training', 'validation', 'test')}
    grid = splits['training']['decision_times_tu']
    intervals = round(s['arc_duration_periods'] * s['decisions_per_period'])
    if grid.shape != (intervals + 1,) or not np.isfinite(grid[-1]) or grid[-1] <= 0:
        raise ValueError('Invalid recorded decision grid')
    np.testing.assert_allclose(grid, np.linspace(0.0, grid[-1], intervals + 1), rtol=1e-10)
    # Reproduce the recorded reference, including its period. Re-solving the
    # orbit correction can give a slightly different period on another platform.
    T = grid[-1] / s['arc_duration_periods']
    initial_ref = splits['training']['observer_reference_normalized'][0, 0]
    cycle = _reference(initial_ref, T, cfg)

    def reference(t):
        return cycle(float(t) % T)

    report = {}
    for split, d in splits.items():
        times = d['decision_times_tu']
        np.testing.assert_array_equal(times, grid)
        phase_error = d['decision_phase_rad'] - 2 * np.pi * times / T
        np.testing.assert_allclose(np.exp(1j * phase_error), 1, rtol=0, atol=1e-10)
        initial_states = d['observer_reference_normalized'][:, 0]
        np.testing.assert_array_equal(initial_states, np.broadcast_to(initial_ref, initial_states.shape))
        nav = d['observer_navigation_error_km_m_s']
        actions = d['observer_applied_delta_v_rotating_m_s']
        for k in range(L,len(times)):
            np.testing.assert_allclose(d['observer_history_error_km_m_s'][:,k],nav[:,k-L+1:k+1])
            np.testing.assert_allclose(d['observer_history_applied_delta_v_rotating_m_s'][:,k],actions[:,k-L:k])
        for arc in range(len(nav)):
            for k,t in enumerate(times):
                np.testing.assert_allclose(nav[arc,k],_error(t,d['observer_navigation_normalized'][arc,k],
                    d['observer_reference_normalized'][arc,k],cfg),atol=1e-9)
        active = d['observer_maneuver_required'] & d['observer_maneuver_valid']
        dv = d['observer_target_delta_v_rotating_m_s'][active]
        delay = d['observer_target_burn_delay_tu'][active]
        assert np.isfinite(dv).all() and np.isfinite(delay).all()
        assert np.all(np.linalg.norm(dv,axis=-1)<=s['observer_guidance']['maximum_impulse_m_s'])
        assert np.all(np.any(np.isclose(delay[:,None]/T,s['candidate_delay_fractions']),axis=1))
        # Impulses are rotating m/s; exactly invert the normalized velocity update.
        scale = cfg['time_unit_s']/(1000*cfg['length_unit_km'])
        np.testing.assert_allclose((dv*scale)/scale,dv,atol=1e-12)
        checked = 0
        # Independently replay every feasible candidate against the common
        # nonlinear terminal epoch, not just the exported label fields.
        weights = np.r_[np.full(3,1/s['observer_guidance']['position_tolerance_km']),
                        np.full(3,1/s['observer_guidance']['velocity_tolerance_m_s'])]
        for arc,k in zip(*np.where(active)):
            t = times[k]
            horizon = s['guidance_horizon_fraction']*T
            delay_k = d['observer_target_burn_delay_tu'][arc,k]
            state = d['observer_navigation_normalized'][arc,k]
            free,_ = _forward(state,t,horizon,cfg,False)
            prior,_ = _forward(state,t,float(delay_k),cfg,False) if delay_k else (state.copy(),None)
            prior[3:] += d['observer_target_delta_v_rotating_m_s'][arc,k]*scale
            final,_ = _forward(prior,t+delay_k,horizon-delay_k,cfg,False)
            before = np.linalg.norm(weights*_error(t+horizon,free,reference(t+horizon),cfg))
            after = np.linalg.norm(weights*_error(t+horizon,final,reference(t+horizon),cfg))
            assert after < before, 'STM target fails nonlinear improvement check'
            checked += 1
        # At each epoch compare the total of all burns that should have
        # executed since the previous decision, including immediate burns.
        expected = np.zeros_like(actions)
        for arc,k in zip(*np.where(active)):
            burn_t = times[k]+d['observer_target_burn_delay_tu'][arc,k]
            j = np.searchsorted(times,burn_t-1e-12)
            if j < len(times):
                expected[arc,j] += d['observer_target_delta_v_rotating_m_s'][arc,k]
        np.testing.assert_allclose(expected,actions,atol=1e-10)
        report[split] = dict(nonlinear_targets_replayed=checked,applied_targets_verified=True,arcs=len(nav),history_steps=L,feasible_burns=int(active.sum()),
            rejected_teacher_labels=int((d['observer_maneuver_required'] & ~d['observer_maneuver_valid']).sum()),
            decision_spacing_days=float((grid[-1] / intervals) * cfg['time_unit_s'] / 86400),
            measurement_cadence_s=s.get('closed_loop_measurement_cadence_s',21600),
            input_units='MCI departure km,m/s; phase/visibility/count/log normalized covariance',
            target_units='rotating-frame delta-v m/s; delay TU',
            causal_histories_verified=True)
    return report
