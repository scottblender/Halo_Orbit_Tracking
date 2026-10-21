"""Spherical Moon, synchronous rotation, and geometric elevation mask."""
import numpy as np
from .frames import rotation,W

def site_rotating(cfg):
    """Return a Moon-relative site vector fixed in the synodic frame."""
    site=cfg['site']; lat=np.deg2rad(site['latitude_deg'])
    lon=np.deg2rad(site['longitude_east_deg'])+site['prime_meridian_phase_rad']
    return site['radius_km']/cfg['length_unit_km']*np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])

def site_mci(t,cfg):
    """Return site position and velocity in km and km/s."""
    q=site_rotating(cfg); L=cfg['length_unit_km']; T=cfg['time_unit_s']
    return L*rotation(t)@q,L/T*rotation(t)@W@q

def elevation(state,cfg):
    """Return geometric elevation in radians; mask includes lunar horizon."""
    site=site_rotating(cfg)
    rho=state[:3]-[1-cfg['mu'],0,0]-site
    return np.arcsin(np.clip(rho@site/(np.linalg.norm(rho)*np.linalg.norm(site)),-1,1))
