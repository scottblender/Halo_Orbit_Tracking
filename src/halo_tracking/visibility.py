"""Unified optical range, physical occultation, and body-center keep-outs.

Angles are measured between the observer-to-target LOS and observer-to-body
direction. Following the cislunar SDA reference code, the operative threshold
is max(physical finite-target occultation, configured angular exclusion).
"""
import numpy as np

AU_KM = 149597870.7
EARTH_RADIUS_KM = 6378.1366
MOON_RADIUS_KM = 1737.4
SUN_RADIUS_KM = 695700.0
SIDEREAL_YEAR_S = 365.256363004 * 86400.


def sun_position_rotating(time_tu, cfg):
    """Approximate Sun direction from a planar bicircular phase model, in LU."""
    opts = cfg.get('optics', {})
    phase = float(opts.get('sun_initial_phase_rad', 0.))
    motion = 2*np.pi*cfg['time_unit_s']/SIDEREAL_YEAR_S - 1.
    angle = phase + motion*time_tu
    radius = AU_KM/cfg['length_unit_km']
    inclination = np.deg2rad(float(opts.get('sun_inclination_deg', 0.)))
    return radius*np.array([np.cos(angle),
                            np.sin(angle)*np.cos(inclination),
                            np.sin(angle)*np.sin(inclination)])


def body_clear(observer, target, center, radius, exclusion_deg=0.):
    """Apply finite-target limb occultation AND configured center separation.

    Physical tangency is blocked; equality to a larger configured keep-out
    threshold is allowed, matching calc_visibility.m in the space-based repo.
    """
    obs = np.asarray(observer, dtype=float)
    tar = np.asarray(target, dtype=float)
    center = np.asarray(center, dtype=float)
    los = tar - obs
    rho = float(np.linalg.norm(los))
    if rho <= 0:
        raise ValueError('Coincident observer and target')
    body = center - obs
    distance = float(np.linalg.norm(body))
    if distance <= radius:
        return False
    separation = np.arccos(np.clip(np.dot(los, body)/(rho*distance), -1., 1.))
    tangent = np.sqrt((distance-radius)*(distance+radius))
    if rho < distance-radius:
        occultation = -np.inf
    elif rho < tangent:
        c = (distance**2 + rho**2 - radius**2)/(2*distance*rho)
        occultation = np.arccos(np.clip(c, -1., 1.))
    else:
        occultation = np.arcsin(min(radius/distance, 1.))
    # Avoid rejecting mathematical equality from acos roundoff (~1e-16 rad).
    # Only the user-configured boundary receives this floating-point slack;
    # strict physical tangency remains blocked.
    configured = np.deg2rad(exclusion_deg)
    return bool(separation > occultation and
                separation + 4*np.finfo(float).eps >= configured)


def optical_visible(observer, target, time_tu, cfg, link='gateway'):
    """Check slant range and unified Sun/Earth/Moon keep-outs in rotating LU."""
    if link not in ('gateway', 'lso'):
        raise ValueError('link must be gateway or lso')
    opts = cfg.get('optics', {})
    observer = np.asarray(observer, dtype=float)
    target = np.asarray(target, dtype=float)
    rho = np.linalg.norm(target-observer)
    if rho <= 0:
        return False
    limit = opts.get(f'{link}_max_range_km')
    if limit is not None and rho*cfg['length_unit_km'] > float(limit):
        return False
    mu = cfg['mu']
    L = cfg['length_unit_km']
    bodies = [
        (np.array([-mu, 0., 0.]), EARTH_RADIUS_KM/L,
         float(opts.get('earth_exclusion_deg', 10.))),
        (sun_position_rotating(time_tu, cfg), SUN_RADIUS_KM/L,
         float(opts.get('sun_exclusion_deg', 20.))),
    ]
    if link == 'gateway':
        bodies.append((np.array([1-mu, 0., 0.]),
                       float(cfg['site']['radius_km'])/L,
                       float(opts.get('moon_exclusion_deg', 0.))))
    return all(body_clear(observer, target, center, radius, keepout)
               for center, radius, keepout in bodies)
