"""Idealized Moon-centered inertial axes aligned with rotating axes at t=0."""
import numpy as np
from .dynamics import rhs
W = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]])

def rotation(t):
    """Rotate synodic components into the model inertial axes."""
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def state_transform(t, cfg):
    """Jacobian from normalized synodic state to km and km/s in MCI."""
    R=rotation(t); L=cfg['length_unit_km']; T=cfg['time_unit_s']
    return np.block([[L*R,np.zeros((3,3))],[L/T*R@W,L/T*R]])

def to_mci(t, state, cfg):
    """Return Moon-relative inertial position, velocity, acceleration."""
    q=state[:3]-np.array([1-cfg['mu'],0.,0.]); v=state[3:]
    R=rotation(t); L=cfg['length_unit_km']; T=cfg['time_unit_s']
    a=rhs(t,state,cfg['mu'])[3:]
    return L*R@q,L/T*R@(v+W@q),L/T**2*R@(a+2*W@v+W@W@q)

def acceleration_jacobian(t,state,cfg):
    """Differentiate physical MCI acceleration with respect to synodic state."""
    from .dynamics import jacobian
    R=rotation(t); L=cfg['length_unit_km']; T=cfg['time_unit_s']
    A=jacobian(state,cfg['mu'])
    return L/T**2*R@np.c_[A[3:,:3]+W@W,A[3:,3:]+2*W]
