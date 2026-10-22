"""One-Gateway-period estimation with exact versus LSO-estimated observer state."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .dynamics import propagate, propagate_step
from .estimation import initial_covariance
from .frames import rotation, to_mci, state_transform
from .measurements import predict, measurement_jacobian, wrap, ARCSEC
from .study import make_truth
from .network import sensor_configs, observations
from .tracking import relative_angles


def spacecraft_measurement(t, observer, gateway, cfg):
    """Return observer-to-Gateway inertial-axis angles in radians."""
    ro,vo,_=to_mci(t,observer,cfg); rg,vg,_=to_mci(t,gateway,cfg)
    return relative_angles(rg-ro,vg-vo)[:2]


def spacecraft_jacobians(t, observer, gateway, cfg):
    """Return angular Jacobians with respect to normalized observer/target states."""
    vector=rotation(t)@(gateway[:3]-observer[:3])*cfg['length_unit_km']
    x,y,z=vector; q=np.hypot(x,y); d2=vector@vector
    if q<1e-10:raise ValueError('Line-of-sight angular coordinates are singular')
    J=np.array([[-y/q**2,x/q**2,0.],[-x*z/(q*d2),-y*z/(q*d2),q/d2]])
    Hg=np.c_[J@rotation(t)*cfg['length_unit_km'],np.zeros((2,3))]
    return -Hg,Hg


def gateway_visible(observer,gateway,cfg):
    """Test line-segment Earth/Moon occultation; no camera exclusion angles."""
    a=observer[:3]; b=gateway[:3]; direction=b-a; length2=direction@direction
    if length2==0:return False
    for center,radius in [(np.array([-cfg['mu'],0,0]),6378.1363/cfg['length_unit_km']),
                          (np.array([1-cfg['mu'],0,0]),cfg['site']['radius_km']/cfg['length_unit_km'])]:
        fraction=np.clip((center-a)@direction/length2,0,1)
        if np.linalg.norm(a+fraction*direction-center)<=radius:return False
    return True


def update(state,P,H,R,innovation,consider_observer=False):
    """Joseph update; Gateway data cannot refine the observer in consider mode."""
    S=H@P@H.T+R; K=np.linalg.solve(S,H@P).T
    if consider_observer:K[:6,:]=0.
    innovation=innovation.copy();innovation[0]=wrap(innovation[0])
    state=state+K@innovation; B=np.eye(len(state))-K@H
    covariance=B@P@B.T+K@R@K.T
    return state,(covariance+covariance.T)/2


def _filter(truth, cfg, z_lso, z_gateway, exact_observer, nominal=False):
    times=truth['times']; o=truth['truth'][0].copy();g=truth['gateway_truth'][0].copy()
    po=initial_covariance(cfg)
    target_cfg=deepcopy(cfg)
    target_cfg['estimation']['initial_position_sigma_km']=cfg['comparison']['gateway_position_sigma_km']
    target_cfg['estimation']['initial_velocity_sigma_km_s']=cfg['comparison']['gateway_velocity_sigma_km_s']
    pg=initial_covariance(target_cfg)
    if not nominal:
        g+=np.random.default_rng(cfg['simulation']['seed']+20).multivariate_normal(np.zeros(6),pg)
        if not exact_observer:o+=np.random.default_rng(cfg['simulation']['seed']+10).multivariate_normal(np.zeros(6),po)
    x=g if exact_observer else np.r_[o,g]
    P=pg if exact_observer else np.block([[po,np.zeros((6,6))],[np.zeros((6,6)),pg]])
    Rl=np.eye(2)*(cfg['measurement']['angle_sigma_arcsec']*ARCSEC)**2
    Rg=np.eye(2)*(cfg['comparison']['gateway_camera_sigma_arcsec']*ARCSEC)**2
    states=[];covariances=[];observer_states=[];observer_covariances=[];cross=[]
    previous=times[0]
    for k,t in enumerate(times):
        if exact_observer:
            x,F,Q=propagate_step(x,previous,t,cfg['mu'],0.)
            P=F@P@F.T+Q
            o=truth['truth'][k]
            if nominal:x=truth['gateway_truth'][k].copy()
        else:
            o,Fo,Qo=propagate_step(x[:6],previous,t,cfg['mu'],cfg['estimation']['acceleration_psd_normalized'])
            g,Fg,Qg=propagate_step(x[6:],previous,t,cfg['mu'],0.)
            F=np.block([[Fo,np.zeros((6,6))],[np.zeros((6,6)),Fg]])
            Q=np.block([[Qo,np.zeros((6,6))],[np.zeros((6,6)),Qg]])
            x=np.r_[o,g];P=F@P@F.T+Q
            if nominal:x=np.r_[truth['truth'][k],truth['gateway_truth'][k]]
            available = truth.get('sensor_visible', truth['lso_available'][:, None])
            for j, local in enumerate(sensor_configs(cfg)):
                if available[k, j]:
                    H=np.c_[measurement_jacobian(t,x[:6],local,False),np.zeros((2,6))]
                    measured = z_lso[k] if z_lso.ndim == 2 else z_lso[k,j]
                    innovation=np.zeros(2) if nominal else measured-predict(t,x[:6],local,False)
                    x,P=update(x,P,H,Rl,innovation)
            o=x[:6]
        if truth['gateway_available'][k]:
            g=x if exact_observer else x[6:]
            Ho,Hg=spacecraft_jacobians(t,o,g,cfg)
            H=Hg if exact_observer else np.c_[Ho,Hg]
            innovation=np.zeros(2) if nominal else z_gateway[k]-spacecraft_measurement(t,o,g,cfg)
            x,P=update(x,P,H,Rg,innovation,consider_observer=not exact_observer)
        states.append(x.copy() if exact_observer else x[6:].copy())
        covariances.append(P.copy() if exact_observer else P[6:,6:].copy())
        observer_states.append(o.copy() if exact_observer else x[:6].copy())
        observer_covariances.append(np.zeros((6,6)) if exact_observer else P[:6,:6].copy())
        cross.append(np.zeros((6,6)) if exact_observer else P[:6,6:].copy())
        previous=t
    return dict(state=np.array(states),covariance=np.array(covariances),observer_state=np.array(observer_states),
                observer_covariance=np.array(observer_covariances),observer_gateway_covariance=np.array(cross))


def run_gateway_comparison(cfg):
    """Run paired estimates and local covariance forecasts over one Gateway period."""
    cfg=deepcopy(cfg)
    if cfg.get('gateway', {}).get('period_tu', 0) <= 0:
        raise ValueError('Gateway period must be positive')
    cfg['simulation']['periods']=cfg['gateway']['period_tu']/cfg['orbit']['period_tu']
    if cfg['gateway'].get('state') is None:raise ValueError('Gateway reference state is required')
    cfg.setdefault('comparison',dict(gateway_position_sigma_km=1.,gateway_velocity_sigma_km_s=1e-5,gateway_camera_sigma_arcsec=1.))
    truth=make_truth(cfg);times=truth['times'];truth['config']=cfg
    truth['gateway_truth']=propagate(np.array(cfg['gateway']['state']),times,cfg['mu'])
    dt=cfg['simulation']['cadence_s']/cfg['time_unit_s']
    scheduled=np.isclose(times/dt,np.rint(times/dt),rtol=0,atol=1e-9)
    truth['lso_available']=truth['visible']&scheduled
    truth['gateway_available']=np.array([gateway_visible(o,g,cfg) for o,g in zip(truth['truth'],truth['gateway_truth'])])&scheduled
    network = observations(times, truth['truth'], cfg, cfg['simulation']['seed']+30)
    truth.update(network)
    truth['lso_available'] = network['sensor_visible'].any(axis=1)
    z_lso = network['measured_angles_rad']
    gateway_rng=np.random.default_rng(cfg['simulation']['seed']+40)
    z_gateway=np.array([spacecraft_measurement(t,o,g,cfg) for t,o,g in zip(times,truth['truth'],truth['gateway_truth'])])
    z_gateway+=gateway_rng.normal(size=z_gateway.shape)*cfg['comparison']['gateway_camera_sigma_arcsec']*ARCSEC
    for name,exact in [('perfect',True),('lso_estimated',False)]:
        truth[name]=_filter(truth,cfg,z_lso,z_gateway,exact)
        truth[name+'_forecast']=_filter(truth,cfg,z_lso,z_gateway,exact,nominal=True)
    return truth


def comparison_metrics(result):
    """Report Gateway error and forecast uncertainty in physical MCI units."""
    cfg=result['config'];out={}
    for name in ['perfect','lso_estimated']:
        error=[];sigma=[];actual_cov=[]
        for k,t in enumerate(result['times']):
            B=state_transform(t,cfg)
            error.append(B@(result[name]['state'][k]-result['gateway_truth'][k]))
            sigma.append(B@result[name+'_forecast']['covariance'][k]@B.T)
            actual_cov.append(B@result[name]['covariance'][k]@B.T)
        error=np.array(error);sigma=np.array(sigma)
        out[name]=dict(position_error_km=np.linalg.norm(error[:,:3],axis=1),
                       velocity_error_m_s=1000*np.linalg.norm(error[:,3:],axis=1),
                       position_forecast_rss_sigma_km=np.sqrt(np.maximum(np.trace(sigma[:,:3,:3],axis1=1,axis2=2),0)),
                       velocity_forecast_rss_sigma_m_s=1000*np.sqrt(np.maximum(np.trace(sigma[:,3:,3:],axis1=1,axis2=2),0)))
        out[name]['position_rmse_km']=float(np.sqrt(np.mean(out[name]['position_error_km']**2)))
        out[name]['velocity_rmse_m_s']=float(np.sqrt(np.mean(out[name]['velocity_error_m_s']**2)))
    return out


def save_comparison(result,directory):
    """Save paired state/covariance histories and clear numerical conclusions."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    metrics=comparison_metrics(result);summary={}
    arrays={key:result[key] for key in ['times','truth','gateway_truth','lso_available','gateway_available','sensor_visible','sensor_elevation_rad','ideal_angles_rad','measured_angles_rad']}
    for name in ['perfect','lso_estimated']:
        item=metrics[name]
        summary[name]={key:item[key] for key in ['position_rmse_km','velocity_rmse_m_s']}
        for key in ['position_error_km','velocity_error_m_s','position_forecast_rss_sigma_km','velocity_forecast_rss_sigma_m_s']:
            summary[name]['final_'+key]=float(item[key][-1]);arrays[name+'_'+key]=item[key]
        for run in [name,name+'_forecast']:
            for key,value in result[run].items():arrays[run+'_'+key]=value
    summary['assumptions']=dict(duration_observer_periods=float(result['config']['simulation']['periods']),duration_gateway_periods=1.,duration_days=float(result['times'][-1]*result['config']['time_unit_s']/86400),
        lso_observation_epochs=int(result['lso_available'].sum()),lso_observations=int(result['sensor_visible'].sum()),gateway_observations=int(result['gateway_available'].sum()),
        measurements='Angles only on both optical links',gateway_prior=f"Separate per-axis prior: {result['config']['comparison']['gateway_position_sigma_km']:g} km / {1000*result['config']['comparison']['gateway_velocity_sigma_km_s']:g} m/s",
        interpretation='Forecast covariance is local/linearized; RMSE is one paired noise realization, not ensemble performance. Observer-Gateway cross-covariance is propagated; Gateway data do not refine the observer.')
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(result['config'],indent=2)+'\n')
    np.savez_compressed(directory/'comparison.npz',**arrays)
    return summary
