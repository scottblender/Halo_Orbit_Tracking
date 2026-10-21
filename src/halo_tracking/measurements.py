"""Topocentric inertial-axis angles and independently measured angular rates."""
import numpy as np
from .frames import to_mci
from .surface_sensor import site_mci
ARCSEC = np.pi/(180*3600)

def wrap(angle):
    """Wrap an angular residual to [-pi, pi)."""
    return (angle+np.pi)%(2*np.pi)-np.pi

def predict(t,state,cfg,rates=False):
    """Return [alpha, delta] rad, optionally [alpha_dot, delta_dot] rad/s."""
    r,v,_=to_mci(t,state,cfg); rs,vs=site_mci(t,cfg)
    x,y,z=r-rs; vx,vy,vz=v-vs
    q=np.hypot(x,y); d2=x*x+y*y+z*z
    if q<1e-10 or d2<1e-20:
        raise ValueError('Undefined line-of-sight angles at coordinate singularity')
    h=[np.arctan2(y,x),np.arctan2(z,q)]
    if rates:
        h += [(x*vy-y*vx)/q**2,(q*vz-z*(x*vx+y*vy)/q)/d2]
    return np.array(h)

def measurement_jacobian(t,state,cfg,rates=False):
    """Return a centered numerical Jacobian in normalized state coordinates."""
    H=np.empty((4 if rates else 2,6))
    for j in range(6):
        step=1e-7; dx=np.zeros(6); dx[j]=step
        d=predict(t,state+dx,cfg,rates)-predict(t,state-dx,cfg,rates)
        d[0]=wrap(d[0]); H[:,j]=d/(2*step)
    return H

def noise_covariance(cfg,rates=False):
    """Build SI measurement covariance; accept a full same-epoch override."""
    m=cfg['measurement']
    if rates and m['rate_source']!='independent':
        raise ValueError('Differenced rates need temporal correlation modeling; use raw angles instead')
    override=m.get('covariance_override')
    if override is not None:
        R=np.asarray(override,dtype=float)
        expected=4 if rates else 2
        if R.shape==(4,4) and not rates:
            R=R[:2,:2]
        if R.shape!=(expected,expected) or not np.allclose(R,R.T):
            raise ValueError('Invalid measurement covariance dimensions or symmetry')
    else:
        sigma=[m['angle_sigma_arcsec']*ARCSEC]*2
        if rates:
            sigma += [m['angular_rate_sigma_arcsec_s']*ARCSEC]*2
        R=np.diag(np.square(sigma))
    np.linalg.cholesky(R)
    return R

def simulate(times,truth,visible,cfg):
    """Generate one shared four-channel stream for a paired AO/AO+rates study."""
    R=noise_covariance(cfg,True); rng=np.random.default_rng(cfg['simulation']['seed']+1)
    z=np.full((len(times),4),np.nan)
    for k,t in enumerate(times):
        if visible[k]:
            z[k]=predict(t,truth[k],cfg,True)+rng.multivariate_normal(np.zeros(4),R)
            z[k,0]=wrap(z[k,0])
    return z
