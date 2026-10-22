"""Surface sensor configurations and scheduled angular observations."""
from copy import deepcopy
import numpy as np
from .measurements import predict, wrap, ARCSEC
from .surface_sensor import elevation, site_rotating


def sensor_configs(cfg):
    sites = cfg.get('sensors', [cfg['site']])
    output = []
    for site in sites:
        local = deepcopy(cfg)
        local['site'].update(site)
        output.append(local)
    return output


def observations(times, states, cfg, seed):
    """Return per-site visibility, elevation, ideal angles and noisy angles."""
    sites = sensor_configs(cfg)
    ideal = np.empty((len(times), len(sites), 2))
    elev = np.empty((len(times), len(sites)))
    for j, local in enumerate(sites):
        ideal[:, j] = [predict(t, x, local) for t, x in zip(times, states)]
        elev[:, j] = [elevation(x, local) for x in states]
    dt = cfg['simulation']['cadence_s']/cfg['time_unit_s']
    scheduled = np.isclose(times/dt, np.rint(times/dt), rtol=0, atol=1e-9)
    visible = (elev >= np.deg2rad(cfg['site']['minimum_elevation_deg'])) & scheduled[:, None]
    limit = cfg.get('optics', {}).get('lso_max_range_km')
    if limit is not None:
        moon = np.array([1 - cfg['mu'], 0., 0.])
        for j, local in enumerate(sites):
            ranges = np.linalg.norm(
                states[:, :3] - moon - site_rotating(local), axis=1
            ) * cfg['length_unit_km']
            visible[:, j] &= ranges <= limit
    measured = ideal + np.random.default_rng(seed).normal(size=ideal.shape)*cfg['measurement']['angle_sigma_arcsec']*ARCSEC
    measured[:, :, 0] = wrap(measured[:, :, 0])
    measured[~visible] = np.nan
    return dict(ideal_angles_rad=ideal, measured_angles_rad=measured,
                sensor_visible=visible, sensor_elevation_rad=elev)
