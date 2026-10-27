"""Full-year STM-only example and tracking-error/uncertainty post-processing."""
from pathlib import Path
import json
import numpy as np
from .year_study import _observer_reference, _initial_conditions, simulate_one_arc
from .gateway_reference_periodic import gateway_periodic_reference
from .network import sensor_configs


def running_rmse(values):
    """RMS of available 3-D error norms, with no filled-in missing epochs."""
    x=np.asarray(values,dtype=float)
    valid=np.isfinite(x)
    counts=np.cumsum(valid)
    sums=np.cumsum(np.where(valid,x*x,0.))
    out=np.full(x.shape,np.nan)
    out[valid]=np.sqrt(sums[valid]/counts[valid])
    return out


def rolling_rmse(day,values,window_days=30.):
    """RMS over (day-window, day], evaluated only at available error samples."""
    day=np.asarray(day,dtype=float);x=np.asarray(values,dtype=float)
    if day.shape!=x.shape or np.any(np.diff(day)<=0) or window_days<=0:
        raise ValueError('Use matched arrays, increasing times and a positive window')
    valid=np.isfinite(x)
    sums=np.r_[0.,np.cumsum(np.where(valid,x*x,0.))]
    counts=np.r_[0,np.cumsum(valid)]
    first=np.searchsorted(day,day-window_days,side='right')
    end=np.arange(len(day))+1
    count=counts[end]-counts[first]
    out=np.full(x.shape,np.nan)
    use=valid & (count>0)
    out[use]=np.sqrt((sums[end]-sums[first])[use]/count[use])
    return out


def position_uncertainty_km(trace_normalized,length_unit_km):
    """sqrt(trace(P_rr)) in km: estimated RMS, not a confidence envelope."""
    trace=np.asarray(trace_normalized,dtype=float)
    if length_unit_km<=0 or np.any(trace[np.isfinite(trace)]<0):
        raise ValueError('Position covariance trace must be nonnegative; length scale positive')
    return length_unit_km*np.sqrt(trace)


def postprocess_stm(episode,cfg,planned_days,seed):
    day=episode['day']
    arrays={k:v for k,v in episode.items() if isinstance(v,np.ndarray)}
    for label,key in [('observer','observer_navigation_position_error_km'),
                      ('gateway','gateway_tracking_position_error_km')]:
        x=episode[key]
        arrays[label+'_running_position_rmse_km']=running_rmse(x)
        arrays[label+'_30_day_position_rmse_km']=rolling_rmse(day,x)
        arrays[label+'_position_uncertainty_km']=position_uncertainty_km(
            episode[label+'_covariance_trace_normalized'],cfg['length_unit_km'])
    arrays['gateway_running_velocity_rmse_m_s']=running_rmse(episode['gateway_tracking_velocity_error_m_s'])
    arrays['gateway_30_day_velocity_rmse_m_s']=rolling_rmse(day,episode['gateway_tracking_velocity_error_m_s'])
    def rms(key):
        x=episode[key];v=x[np.isfinite(x)]
        return float(np.sqrt(np.mean(v*v))) if len(v) else None
    good=np.isfinite(episode['observer_truth_position_departure_km'])
    summary=dict(planned_days=planned_days,seed=seed,completed_requested_span=episode['completed_full_span'],
        last_simulated_day=float(day[good][-1]) if good.any() else None,
        stop_reason=episode['halted_reason'],evaluated_samples=int(good.sum()),planned_samples=len(day),
        metric_sampling='Decision epochs plus final endpoint; optical measurements every six hours plus decision epochs by default',
        rmse_interpretation='RMS of 3-D error norms at sampled times, not per-axis or continuous-time RMSE',
        uncertainty_interpretation='sqrt(trace(P_rr)) converted from normalized length to km; estimated RMS, not a confidence bound',
        total_delta_v_m_s=float(episode['delta_v_cost_m_s'].sum()),
        maneuvers=int(episode['executed_burn_count'].sum()),
        rmse=dict(halo_position_departure_km=rms('observer_truth_position_departure_km'),
            halo_velocity_departure_m_s=rms('observer_truth_velocity_departure_m_s'),
            halo_navigation_position_km=rms('observer_navigation_position_error_km'),
            halo_navigation_velocity_m_s=rms('observer_navigation_velocity_error_m_s'),
            gateway_position_km=rms('gateway_tracking_position_error_km'),
            gateway_velocity_m_s=rms('gateway_tracking_velocity_error_m_s')),
        uncertainty={})
    limits=cfg['narx']['observer_guidance']
    inside=good & (episode['observer_truth_position_departure_km']<=limits['position_tolerance_km']) & (
        episode['observer_truth_velocity_departure_m_s']<=limits['velocity_tolerance_m_s'])
    summary['halo_corridor_fraction']=float(inside.mean())
    for label in ('observer','gateway'):
        u=arrays[label+'_position_uncertainty_km']
        first=u[(day<=30.) & np.isfinite(u)]
        last=u[(day>planned_days-30.) & np.isfinite(u)]
        start=float(np.median(first)) if len(first) else None
        end=float(np.median(last)) if len(last) else None
        summary['uncertainty'][label]=dict(first_30_days_median_km=start,last_30_days_median_km=end,
            end_to_start_ratio=end/start if start is not None and start>0 and end is not None else None,
            peak_km=float(np.max(u[np.isfinite(u)])) if np.isfinite(u).any() else None)
    return arrays,summary


def run_stm_baseline(cfg,output_dir,days=365.,seed=24017):
    """Simulate STM guidance and EKF tracking; no learned model is used."""
    if days<=0:raise ValueError('days must be positive')
    ref,T,corrected=_observer_reference(cfg)
    gateway,gateway_check=gateway_periodic_reference(cfg)
    s=cfg['narx'];end=days*86400./cfg['time_unit_s']
    cadence=float(s.get('closed_loop_measurement_cadence_s',21600.))
    if cadence<=0:raise ValueError('Measurement cadence must be positive')
    decisions=np.r_[np.arange(0.,end,T/s['decisions_per_period']),end]
    observations=np.unique(np.r_[np.arange(0.,end,cadence/cfg['time_unit_s']),decisions])
    initial=_initial_conditions(cfg,np.asarray(corrected['orbit']['state']),gateway(0),seed,
        len(observations),len(sensor_configs(cfg)))
    episode=simulate_one_arc(cfg,'stm',ref,T,gateway,decisions,observations,initial)
    arrays,summary=postprocess_stm(episode,cfg,days,seed)
    summary['measurement_cadence_s']=cadence
    summary['decisions_per_period']=s['decisions_per_period']
    summary['decision_spacing_days']=T/s['decisions_per_period']*cfg['time_unit_s']/86400.
    summary['gateway_reference_check']=gateway_check
    summary['configuration']=cfg
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'stm_histories.npz',**arrays)
    np.savez_compressed(out/'initial_conditions.npz',truth=initial[0],estimate=initial[1],covariance=initial[2],
        optical_noise=initial[3],camera_noise=initial[4])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return arrays,summary
