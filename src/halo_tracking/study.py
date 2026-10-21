"""Reproducible paired measurement studies and portable result saving."""
import json
import warnings
from pathlib import Path
import numpy as np
from .dynamics import propagate,jacobi
from .surface_sensor import elevation

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
