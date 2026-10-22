"""Gateway estimation with exact or continuously LSO-estimated observer state."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .dynamics import propagate, propagate_step
from .gateway_reference import GatewayReference
from .orbit_reference import correct_halo_reference
from .surface_sensor import elevation
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
    limit = cfg.get('optics', {}).get('gateway_max_range_km')
    if limit is not None and np.sqrt(length2)*cfg['length_unit_km'] > limit:
        return False
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


def run_gateway_comparison(cfg, start_tu=0., observer_initial_state=None,
                           gateway_reference=None, duration_periods=1):
    """Compare tracking over one observer period at the selected epoch."""
    cfg=deepcopy(cfg)
    if cfg['gateway'].get('state') is None:
        raise ValueError('Gateway reference state is required')
    reference = gateway_reference if gateway_reference is not None else GatewayReference(cfg)
    cfg.setdefault('comparison',dict(gateway_position_sigma_km=1.,gateway_velocity_sigma_km_s=1e-5,gateway_camera_sigma_arcsec=1.))
    if duration_periods < 1 or int(duration_periods) != duration_periods:
        raise ValueError('duration_periods must be a positive integer')
    cfg['simulation']['periods']=int(duration_periods)
    if start_tu == 0. and observer_initial_state is None:
        truth=make_truth(cfg)
        times=truth['times']
    else:
        if start_tu < 0 or observer_initial_state is None:
            raise ValueError('Supply a nonnegative start and observer initial state')
        step=cfg['simulation']['cadence_s']/cfg['time_unit_s']
        end=start_tu+duration_periods*cfg['orbit']['period_tu']
        schedule=np.arange(np.ceil(start_tu/step)*step,end,step)
        boundaries=start_tu+np.arange(1, int(duration_periods))*cfg['orbit']['period_tu']
        times=np.unique(np.r_[start_tu,schedule,boundaries,end])
        observer=propagate(np.asarray(observer_initial_state),times,cfg['mu'])
        elev=np.array([elevation(x,cfg) for x in observer])
        truth=dict(times=times,truth=observer,elevation=elev,
                   visible=elev>=np.deg2rad(cfg['site']['minimum_elevation_deg']))
    truth['config']=cfg
    truth['gateway_truth']=reference.states(times)
    truth['gateway_reference_closure']=reference.closure()
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
    summary['gateway_reference_closure']=result['gateway_reference_closure']
    arrays={key:result[key] for key in ['times','truth','gateway_truth','lso_available','gateway_available','sensor_visible','sensor_elevation_rad','ideal_angles_rad','measured_angles_rad']}
    for name in ['perfect','lso_estimated']:
        item=metrics[name]
        summary[name]={key:item[key] for key in ['position_rmse_km','velocity_rmse_m_s']}
        for key in ['position_error_km','velocity_error_m_s','position_forecast_rss_sigma_km','velocity_forecast_rss_sigma_m_s']:
            summary[name]['final_'+key]=float(item[key][-1]);arrays[name+'_'+key]=item[key]
        for run in [name,name+'_forecast']:
            for key,value in result[run].items():arrays[run+'_'+key]=value
    summary['assumptions']=dict(duration_observer_periods=float(result['config']['simulation']['periods']),duration_gateway_periods=float(result['config']['orbit']['period_tu']/result['config']['gateway']['period_tu']),duration_days=float((result['times'][-1]-result['times'][0])*result['config']['time_unit_s']/86400),
        lso_observation_epochs=int(result['lso_available'].sum()),lso_observations=int(result['sensor_visible'].sum()),gateway_observations=int(result['gateway_available'].sum()),
        measurements='Angles only on both optical links',gateway_prior=f"Separate per-axis prior: {result['config']['comparison']['gateway_position_sigma_km']:g} km / {1000*result['config']['comparison']['gateway_velocity_sigma_km_s']:g} m/s",
        interpretation='Forecast covariance is local/linearized; RMSE is one paired noise realization, not ensemble performance. Observer-Gateway cross-covariance is propagated; Gateway data do not refine the observer.')
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(result['config'],indent=2)+'\n')
    np.savez_compressed(directory/'comparison.npz',**arrays)
    return summary


def run_gateway_period_study(cfg, periods=20):
    """Track Gateway through uninterrupted observer departure, without filter resets.

    The physical observer begins with a deterministic 1-km, 0.01-m/s
    perturbation of the corrected periodic reference. All filter states and
    covariances carry forward across observer-period boundaries.
    """
    if periods < 1 or int(periods) != periods:
        raise ValueError('periods must be a positive integer')
    count = int(periods)
    corrected = correct_halo_reference(cfg)
    observer_period = corrected['orbit']['period_tu']
    initial = np.asarray(corrected['orbit']['state'], dtype=float)
    noise = np.random.default_rng(corrected['simulation']['seed'] + 921).normal(size=6)
    scale = np.array([1. / corrected['length_unit_km']]*3 +
                     [0.01 * .001 * corrected['time_unit_s'] /
                      corrected['length_unit_km']]*3)
    initial_truth = initial + noise * scale
    full = run_gateway_comparison(
        corrected, start_tu=0., observer_initial_state=initial_truth,
        duration_periods=count)
    metrics = comparison_metrics(full)
    times = full['times']
    # Assign boundaries to the new period except the final endpoint.
    ids = np.minimum(np.floor(times / observer_period + 1e-10).astype(int),
                     count - 1)
    reference = np.asarray(corrected['orbit']['state'])
    ranges = np.linalg.norm(
        (full['truth'][:, :3] - full['gateway_truth'][:, :3])
        * corrected['length_unit_km'], axis=1)
    rows, baselines = [], {}
    for k in range(count):
        mask = ids == k
        if not np.any(mask):
            raise ValueError(f'No samples available for period {k+1}')
        start = k*observer_period
        at_start = np.argmin(np.abs(times - start))
        B = state_transform(times[at_start], corrected)
        offset = B @ (full['truth'][at_start] - reference)
        departure = float(np.linalg.norm(offset[:3]))
        for name in ['perfect', 'lso_estimated']:
            item = metrics[name]
            position = float(np.sqrt(np.mean(item['position_error_km'][mask]**2)))
            velocity = float(np.sqrt(np.mean(item['velocity_error_m_s'][mask]**2)))
            baselines.setdefault(name, position)
            baseline = baselines[name]
            change = 100*(position/baseline - 1) if baseline > 0 else float('nan')
            rows.append(dict(
                observer_period=k+1,
                observer_departure_km=departure,
                case=name,
                gateway_observations=int(full['gateway_available'][mask].sum()),
                lso_observations=int(full['sensor_visible'][mask].sum()),
                mean_gateway_range_km=float(np.mean(ranges[mask])),
                position_rmse_km=position,
                velocity_rmse_m_s=velocity,
                final_position_forecast_rss_sigma_km=float(
                    item['position_forecast_rss_sigma_km'][mask][-1]),
                position_rmse_change_pct=float(change)))
    return dict(rows=rows, reference_closure=full['gateway_reference_closure'],
                observer_period_tu=observer_period,
                tracking_window_tu=observer_period,
                periods=count, config_signature=_config_signature(cfg),
                method='continuous_corrected_reference_v1')



def _config_signature(cfg):
    """Fingerprint configuration inputs to avoid reusing stale Gateway metrics."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def load_gateway_period_study(cfg, directory, periods=20):
    """Reuse CSV metrics only when period count, window and configuration agree."""
    directory = Path(directory)
    csv_path = directory / 'gateway_period_metrics.csv'
    summary_path = directory / 'gateway_period_summary.json'
    if not csv_path.is_file() or not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        if (summary.get('method') != 'continuous_corrected_reference_v1'
                or summary.get('config_signature') != _config_signature(cfg)
                or summary.get('periods') != int(periods)
                or not np.isclose(summary.get('tracking_window_tu', np.nan),
                                  correct_halo_reference(cfg)['orbit']['period_tu'])):
            return None
        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'observer_period', 'observer_departure_km', 'case',
                        'gateway_observations', 'lso_observations',
                        'mean_gateway_range_km', 'position_rmse_km',
                        'velocity_rmse_m_s', 'final_position_forecast_rss_sigma_km',
                        'position_rmse_change_pct'}
            if not required.issubset(reader.fieldnames or []):
                return None
            rows = list(reader)
        if len(rows) != 2 * int(periods):
            return None
        for i, item in enumerate(rows):
            item['observer_period'] = int(item['observer_period'])
            for key in ['gateway_observations', 'lso_observations']:
                item[key] = int(item[key])
            for key in (required - {'observer_period', 'case',
                                    'gateway_observations', 'lso_observations'}):
                item[key] = float(item[key])
            if (item['observer_period'] != i // 2 + 1
                    or item['case'] != ('perfect' if i % 2 == 0 else 'lso_estimated')):
                return None
        return dict(rows=rows, reference_closure=summary['reference_closure'],
                    observer_period_tu=float(summary['observer_period_tu']),
                    tracking_window_tu=float(summary['tracking_window_tu']),
                    periods=int(periods), config_signature=summary['config_signature'],
                    method=summary['method'])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_gateway_period_study(study, directory):
    """Save comparable per-observer-period Gateway tracking results."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'gateway_period_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(study['rows'][0]))
        writer.writeheader()
        writer.writerows(study['rows'])
    (directory/'gateway_period_summary.json').write_text(
        json.dumps(dict(reference_closure=study['reference_closure'],
                        observer_period_tu=study['observer_period_tu'],
                        tracking_window_tu=study['tracking_window_tu'],
                        periods=study['periods'],
                        config_signature=study['config_signature'],
                        method=study['method'],
                        interpretation=('One continuous tracking run spans all observer periods; '
                                        'filter states/covariances are never reset between periods. '
                                        'Truth observer begins with a seeded 1 km and 0.01 m/s '
                                        'per-axis perturbation of the differentially corrected '
                                        'halo reference. Optical range gates are illustrative, '
                                        'not calibrated instrument detection models. RMSE is '
                                        'from one noise realization.')),indent=2)+'\n')
