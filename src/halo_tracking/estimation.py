"""Known-orbit EKF with Joseph updates, wrapped residuals, and visibility gaps."""
import numpy as np
from .dynamics import propagate_step
from .measurements import predict,measurement_jacobian,noise_covariance,wrap

def initial_covariance(cfg):
    """Create a diagonal prior in normalized rotating position and velocity."""
    e=cfg['estimation']; L=cfg['length_unit_km']; T=cfg['time_unit_s']
    return np.diag([ (e['initial_position_sigma_km']/L)**2]*3+[(e['initial_velocity_sigma_km_s']*T/L)**2]*3)

def run_ekf(times,truth,visible,measurements,cfg,rates=False,covariance_only=False):
    """Run a six-state EKF; nominal covariance mode uses zero innovations."""
    P=initial_covariance(cfg); rng=np.random.default_rng(cfg['simulation']['seed'])
    x=truth[0].copy()
    if not covariance_only:
        x += rng.multivariate_normal(np.zeros(6),P)
    R=noise_covariance(cfg,rates); m=len(R); I=np.eye(6)
    xs=[]; Ps=[]; nis=np.full(len(times),np.nan); innovations=np.full((len(times),m),np.nan)
    previous=times[0]
    for k,t in enumerate(times):
        x,F,Q=propagate_step(x,previous,t,cfg['mu'],cfg['estimation']['acceleration_psd_normalized'])
        P=F@P@F.T+Q; P=(P+P.T)/2
        if covariance_only:
            x=truth[k].copy()
        if visible[k]:
            H=measurement_jacobian(t,x,cfg,rates)
            S=H@P@H.T+R
            K=np.linalg.solve(S,H@P).T
            innovation=np.zeros(m) if covariance_only else measurements[k,:m]-predict(t,x,cfg,rates)
            innovation[0]=wrap(innovation[0]); x=x+K@innovation
            B=I-K@H; P=B@P@B.T+K@R@K.T
            innovations[k]=innovation; nis[k]=innovation@np.linalg.solve(S,innovation)
        xs.append(x.copy()); Ps.append((P+P.T)/2); previous=t
    return dict(state=np.array(xs),covariance=np.array(Ps),innovations=innovations,nis=nis,
                rates=rates,covariance_only=covariance_only)
