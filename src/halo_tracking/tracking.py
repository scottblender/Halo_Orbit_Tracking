"""Map uncertain observer position/velocity into Gateway optical geometry."""
import numpy as np
from .dynamics import propagate
from .frames import to_mci, state_transform
from .measurements import ARCSEC, wrap


def relative_angles(relative_position,relative_velocity):
    """Return alpha/delta in rad and their rates in rad/s."""
    x,y,z=relative_position; vx,vy,vz=relative_velocity
    horizontal=np.hypot(x,y); squared_range=relative_position@relative_position
    if horizontal<1e-10 or squared_range<1e-20:
        raise ValueError('Observer-target angular coordinates are singular')
    return np.array([np.arctan2(y,x),np.arctan2(z,horizontal),
                     (x*vy-y*vx)/horizontal**2,
                     (horizontal*vz-z*(x*vx+y*vy)/horizontal)/squared_range])


def observer_tracking_covariance(t,observer_state,observer_covariance,target_state,cfg):
    """Return observer-induced angular/rate covariance for a known target.

    Inputs are normalized synodic states and observer covariance. The target
    is treated as exactly known; no target/observer cross-covariance is assumed.
    Tangent-plane pointing uses a nonsingular local two-axis representation.
    """
    r,v,_=to_mci(t,observer_state,cfg)
    rt,vt,_=to_mci(t,target_state,cfg)
    relative_position=rt-r; relative_velocity=vt-v
    distance=np.linalg.norm(relative_position)
    u=relative_position/distance
    reference=np.eye(3)[np.argmin(np.abs(u))]
    first=np.cross(reference,u); first/=np.linalg.norm(first)
    basis=np.vstack([first,np.cross(u,first)])
    C=state_transform(t,cfg)@observer_covariance@state_transform(t,cfg).T
    J=-(np.eye(3)-np.outer(u,u))/distance
    pointing=basis@J@C[:3,:3]@J.T@basis.T
    H=np.empty((4,6))
    for j in range(6):
        step=1e-7; delta=np.zeros(6);delta[j]=step
        plus=to_mci(t,observer_state+delta,cfg)
        minus=to_mci(t,observer_state-delta,cfg)
        difference=relative_angles(rt-plus[0],vt-plus[1])-relative_angles(rt-minus[0],vt-minus[1])
        difference[0]=wrap(difference[0]); H[:,j]=difference/(2*step)
    angular=H@observer_covariance@H.T
    return dict(range_km=float(distance),pointing_covariance_rad2=pointing,
                pointing_rss_sigma_arcsec=float(np.sqrt(max(np.trace(pointing),0))/ARCSEC),
                angular_measurement_covariance=angular,
                angular_rate_rss_sigma_arcsec_s=float(np.sqrt(max(np.trace(angular[2:,2:]),0))/ARCSEC),
                range_sigma_km=float(np.sqrt(max(u@C[:3,:3]@u,0))))


def gateway_tracking_history(result):
    """Project both observer covariance cases through a known Gateway state."""
    cfg=result['config']; gateway=cfg['gateway']
    if gateway.get('state') is None:
        raise ValueError('Provide a normalized Gateway reference state before computing tracking impact')
    target=propagate(np.asarray(gateway['state']),result['times'],cfg['mu'])
    output=dict(times=result['times'],gateway_state=target)
    for name in ['angles','angles_rates']:
        entries=[observer_tracking_covariance(t,x,P,g,cfg) for t,x,P,g in
                 zip(result['times'],result['truth'],result[name]['covariance'],target)]
        output[name]={key:np.array([entry[key] for entry in entries]) for key in entries[0]}
    return output
