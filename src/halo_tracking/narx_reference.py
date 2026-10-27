"""One neural NARX versus the STM reference: burns, phase, and orbit error."""
import json
from pathlib import Path
import numpy as np
import torch
from .neural_narx import fit_neural_policy, exogenous, targets, sequence
from .two_stage_narx import _load
from .year_study import _observer_reference, _initial_conditions, simulate_one_arc
from .gateway_reference_periodic import gateway_periodic_reference
from .network import sensor_configs


def circular_phase_error(predicted_deg, reference_deg):
    """Signed shortest phase difference, in [-180,180) degrees."""
    return (np.asarray(predicted_deg)-np.asarray(reference_deg)+180.) % 360.-180.


def decode_predictions(raw, maximum):
    """One network: burn score, three dv outputs, and delay candidate scores."""
    requested = raw[...,0] > .5
    valid = np.isfinite(raw).all(-1) & (np.linalg.norm(raw[...,1:4],axis=-1)<=maximum)
    accepted = requested & valid
    dv = np.where(accepted[...,None],raw[...,1:4],0.)
    return dv,accepted,requested & ~valid


def fit_reference_narx(dataset_dir,cfg,output_dir,epochs=600,hidden=64):
    """Train ONE network and save held-out reference/prediction residuals."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    bundle,training_report=fit_neural_policy(dataset_dir,cfg,out/'model',epochs=epochs,hidden=hidden)
    data={s:_load(Path(dataset_dir)/(s+'.npz')) for s in ('training','test')}
    labels={s:targets(d,cfg) for s,d in data.items()}
    d=data['test']; x=exogenous(d); truth=labels['test']
    L=bundle['history_steps']
    known=~(d['observer_maneuver_required'] & ~d['observer_maneuver_valid'])
    mask=np.isfinite(x).all(-1) & known
    mask[:,:L]=False
    maximum=cfg['narx']['observer_guidance']['maximum_impulse_m_s']
    with torch.no_grad():
        p,_=sequence(bundle,x,truth,False)
        raw=p.numpy()*bundle['y_scale']+bundle['y_mean']
        # Same trained weights, diagnostic only: correct past reference outputs.
        p,_=sequence(bundle,x,truth,True)
        with_reference_history=p.numpy()*bundle['y_scale']+bundle['y_mean']
    predicted,predicted_burn,rejected=decode_predictions(raw,maximum)
    correct_history,_,_=decode_predictions(with_reference_history,maximum)
    reference=truth[...,1:4]
    reference_burn=truth[...,0]>.5
    active=mask & reference_burn
    residual=predicted-reference
    period=float(d['decision_times_tu'][-1])/cfg['narx']['arc_duration_periods']
    delay=np.asarray(bundle['candidate_delay_fractions'])[raw[...,4:].argmax(-1)]*period
    predicted_phase=(360*(d['decision_times_tu'][None,:]+delay)/period) % 360
    reference_phase=np.rad2deg(d['observer_target_burn_phase_rad']) % 360
    phase_residual=circular_phase_error(predicted_phase,reference_phase)
    phase_residual[~active]=np.nan
    def vector_rmse(error,rows):
        return float(np.sqrt(np.mean(np.sum(error[rows]**2,axis=-1)))) if rows.any() else None
    train_burns=labels['training'][:,L:,0]>.5
    stats=dict(epochs_trained=epochs,selected_epoch=training_report['selected_epoch'],
        trained_networks=1,training_decisions=int(train_burns.size),training_burns=int(train_burns.sum()),
        test_decisions=int(mask.sum()),reference_burns=int(active.sum()),
        predicted_burns=int((mask & predicted_burn).sum()),
        missed_burns=int((active & ~predicted_burn).sum()),
        extra_burns=int((mask & ~reference_burn & predicted_burn).sum()),
        rejected_predictions=int((mask & rejected).sum()),
        vector_rmse_m_s=vector_rmse(residual,mask),
        burn_vector_rmse_m_s=vector_rmse(residual,active),
        correct_history_burn_vector_rmse_m_s=vector_rmse(correct_history-reference,active),
        component_bias_m_s=residual[mask].mean(0).tolist(),
        phase_mean_absolute_error_deg=float(np.mean(np.abs(phase_residual[active]))) if active.any() else None,
        phase_exact_candidate_accuracy=float(np.mean(np.abs(phase_residual[active])<1e-6)) if active.any() else None,
        phase_evaluation='Timing estimates at all reference burn decisions; a timing estimate alone does not mean a burn was requested.',
        phase_candidates_relative_deg=(360*np.asarray(bundle['candidate_delay_fractions'])).tolist(),
        training_timing_counts=np.bincount(labels['training'][:,L:,4:][train_burns].argmax(-1),minlength=len(bundle['candidate_delay_fractions'])).tolist(),
        test_timing_counts=np.bincount(truth[...,4:][active].argmax(-1),minlength=len(bundle['candidate_delay_fractions'])).tolist())
    np.savez_compressed(out/'prediction_residuals.npz',reference_delta_v=reference,
        predicted_delta_v=predicted,delta_v_residual=residual,raw_network_output=raw,
        reference_phase_deg=reference_phase,predicted_phase_deg=predicted_phase,
        phase_residual_deg=phase_residual,reference_burn=reference_burn,
        predicted_burn=predicted_burn,mask=mask,day=d['decision_times_tu']*cfg['time_unit_s']/86400)
    (out/'prediction_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    return bundle,stats


def run_reference_pair(cfg,bundle,output_dir,days=365.,seed=24017):
    """Apply STM and NARX burns in two matched nonlinear simulations.

    The same initial states and epoch-indexed sensor noise are used. The NARX
    run never calls STM guidance and never uses reference burns as feedback.
    """
    s=cfg['narx']; trained=bundle['configuration']
    for key in ('history_steps','decisions_per_period','candidate_delay_fractions','observer_guidance'):
        if trained[key]!=s[key]:
            raise ValueError('Training/deployment mismatch: '+key)
    cadence=float(s.get('closed_loop_measurement_cadence_s',21600.))
    if cadence!=float(trained.get('closed_loop_measurement_cadence_s',21600.)):
        raise ValueError('Training/deployment measurement cadence mismatch')
    if days<=0: raise ValueError('days must be positive')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    observer,T,corrected=_observer_reference(cfg)
    gateway,_=gateway_periodic_reference(cfg)
    end=days*86400./cfg['time_unit_s']
    grid=np.r_[np.arange(0.,end,T/s['decisions_per_period']),end]
    observations=np.unique(np.r_[np.arange(0.,end,cadence/cfg['time_unit_s']),grid])
    initial=_initial_conditions(cfg,np.asarray(corrected['orbit']['state']),gateway(0),
        seed,len(observations),len(sensor_configs(cfg)))
    np.savez_compressed(out/'paired_conditions.npz',truth=initial[0],estimate=initial[1],
        covariance=initial[2],optical_noise=initial[3],camera_noise=initial[4])
    episodes={}; summary={}
    for label,controller in [('STM','stm'),('NARX','neural_narx')]:
        e=simulate_one_arc(cfg,controller,observer,T,gateway,grid,observations,initial,model=bundle)
        episodes[label]=e
        np.savez_compressed(out/(label.lower()+'_orbit.npz'),**{k:v for k,v in e.items() if isinstance(v,np.ndarray)})
        p=e['observer_truth_position_departure_km']; v=e['observer_truth_velocity_departure_m_s']
        valid=np.isfinite(p) & np.isfinite(v)
        if not valid.any(): raise RuntimeError(label+' simulation produced no evaluated states')
        summary[label]=dict(completed_365_days=e['completed_full_span'] if days==365 else None,
            completed_requested_span=e['completed_full_span'],last_simulated_day=float(e['day'][valid][-1]),
            maximum_position_departure_km=float(np.max(p[valid])),
            maximum_velocity_departure_m_s=float(np.max(v[valid])),
            corridor_fraction=float(np.mean(valid & (p<=s['observer_guidance']['position_tolerance_km'])
                & (v<=s['observer_guidance']['velocity_tolerance_m_s']))),
            total_delta_v_m_s=float(e['delta_v_cost_m_s'].sum()),
            stop_reason=e['halted_reason'],rejected_burns=int(e['maneuver_rejected'].sum()))
        print(label+': '+json.dumps(summary[label]),flush=True)
    # Report growing deployment inputs without claiming it alone caused failure.
    z=episodes['NARX']['neural_input_max_abs_z']
    finite=np.isfinite(z)
    summary['diagnostics']=dict(maximum_NARX_input_standard_deviations=float(np.max(z[finite])) if finite.any() else None,
        first_day_above_10_standard_deviations=float(episodes['NARX']['day'][finite & (z>10)][0]) if np.any(finite & (z>10)) else None,
        note='Large values indicate inputs far from training scaling; this is not a calibrated uncertainty or causal proof.')
    (out/'orbit_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    return episodes,summary
