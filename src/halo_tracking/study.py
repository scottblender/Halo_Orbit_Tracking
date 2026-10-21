"""Reproducible paired measurement studies and portable result saving."""
import json
import warnings
from pathlib import Path
import numpy as np
from .dynamics import propagate,jacobi
from .surface_sensor import elevation
from .measurements import simulate
from .estimation import run_ekf
from .frames import to_mci,state_transform,acceleration_jacobian

def load_config(path=None):
    """Load the editable baseline configuration from a source checkout."""
    if path is None:
        path=Path(__file__).resolve().parents[2]/'configs'/'baseline.json'
    cfg=json.loads(Path(path).read_text())
    if len(cfg['orbit']['state'])!=6 or not np.all(np.isfinite(cfg['orbit']['state'])):
        raise ValueError('Orbit state must contain six finite components')
    if not 0<cfg['mu']<.5 or cfg['time_unit_s']<=0 or cfg['length_unit_km']<=0:
        raise ValueError('Invalid normalization constants')
    if cfg['simulation']['periods']<=0 or cfg['simulation']['cadence_s']<=0 or cfg['orbit']['period_tu']<=0:
        raise ValueError('Duration, cadence and nominal period must be positive')
    if cfg['estimation']['acceleration_psd_normalized']<0:
        raise ValueError('Acceleration PSD must be nonnegative')
    return cfg

def make_truth(cfg, extra_times=None):
    """Propagate the supplied rounded state, preserving its catalog identity."""
    end=cfg['simulation']['periods']*cfg['orbit']['period_tu']
    dt=cfg['simulation']['cadence_s']/cfg['time_unit_s']
    extra = np.asarray([] if extra_times is None else extra_times, dtype=float)
    if np.any(~np.isfinite(extra)) or np.any(extra<0) or np.any(extra>end):
        raise ValueError('Extra epochs must lie within the propagation span')
    times=np.unique(np.r_[np.arange(0,end,dt),end,extra])
    truth=propagate(np.array(cfg['orbit']['state']),times,cfg['mu'])
    elev=np.array([elevation(x,cfg) for x in truth])
    visible=elev>=np.deg2rad(cfg['site']['minimum_elevation_deg'])
    period_state=propagate(np.array(cfg['orbit']['state']),[0,cfg['orbit']['period_tu']],cfg['mu'])[-1]
    closure=period_state-np.array(cfg['orbit']['state'])
    diagnostics=dict(period_closure_position_km=float(np.linalg.norm(closure[:3])*cfg['length_unit_km']),
        period_closure_velocity_km_s=float(np.linalg.norm(closure[3:])*cfg['length_unit_km']/cfg['time_unit_s']),
        jacobi_range=float(np.ptp([jacobi(x,cfg['mu']) for x in truth])),visible_epochs=int(visible.sum()),total_epochs=len(times))
    if diagnostics['period_closure_position_km']>1:
        warnings.warn('Supplied rounded state does not close within 1 km. Treat as an initial-value trajectory, not a validated periodic halo.',stacklevel=2)
    if not visible.any():
        warnings.warn('No visible epochs: estimation will be prior-only propagation.',stacklevel=2)
    return dict(times=times,truth=truth,elevation=elev,visible=visible,diagnostics=diagnostics)

def run_study(cfg=None,covariance_only=False):
    """Compare AO and AO+independent rates on the same truth and noise stream."""
    cfg=load_config() if cfg is None else cfg
    result=make_truth(cfg); result['config']=cfg
    z=None if covariance_only else simulate(result['times'],result['truth'],result['visible'],cfg)
    result['measurements']=z
    for name,rates in [('angles',False),('angles_rates',True)]:
        result[name]=run_ekf(result['times'],result['truth'],result['visible'],z,cfg,rates,covariance_only)
    return result

def physical_history(result,name):
    """Return MCI kinematics, errors, covariance and derived acceleration sigma."""
    cfg=result['config']; out=result[name]; values=[]; errors=[]; covariances=[]; asig=[]
    for t,truth,x,P in zip(result['times'],result['truth'],out['state'],out['covariance']):
        kin=np.concatenate(to_mci(t,x,cfg)); ref=np.concatenate(to_mci(t,truth,cfg))
        B=state_transform(t,cfg); G=acceleration_jacobian(t,x,cfg)
        values.append(kin); errors.append(kin-ref); covariances.append(B@P@B.T)
        asig.append(np.sqrt(np.maximum(np.diag(G@P@G.T),0)))
    return dict(values=np.array(values),errors=np.array(errors),covariance=np.array(covariances),acceleration_sigma=np.array(asig))

def summary(result):
    """Report errors and uncertainty without claiming estimator optimality."""
    table={}
    for name in ['angles','angles_rates']:
        p=physical_history(result,name); e=p['errors']; P=p['covariance']
        table[name]=dict(position_rmse_km=float(np.sqrt(np.mean(np.sum(e[:,:3]**2,axis=1)))),
            velocity_rmse_km_s=float(np.sqrt(np.mean(np.sum(e[:,3:6]**2,axis=1)))),
            final_position_rss_sigma_km=float(np.sqrt(np.trace(P[-1,:3,:3]))),
            mean_nis=float(np.nanmean(result[name]['nis'])) if result['visible'].any() else None)
    return table

def save_study(result,directory):
    """Save complete configuration, diagnostics, and numerical histories."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    (directory/'config.json').write_text(json.dumps(result['config'],indent=2))
    (directory/'summary.json').write_text(json.dumps(dict(diagnostics=result['diagnostics'],metrics=summary(result)),indent=2))
    arrays={k:result[k] for k in ['times','truth','elevation','visible']}
    if result['measurements'] is not None:
        arrays['measurements']=result['measurements']
    for name in ['angles','angles_rates']:
        for key in ['state','covariance','innovations','nis']:
            arrays[name+'_'+key]=result[name][key]
    np.savez_compressed(directory/'study.npz',**arrays)
