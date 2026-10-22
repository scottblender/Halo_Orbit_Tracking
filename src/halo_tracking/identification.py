"""Generate independent one-period arcs for local dynamics identification."""
from pathlib import Path
import json
import numpy as np
from .dynamics import propagate
from .frames import to_mci
from .network import observations


def generate_dataset(cfg, directory):
    """Save trajectory-level train/validation/test splits without fitting a model."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    setup = cfg['identification']
    step = cfg['simulation']['cadence_s']/cfg['time_unit_s']
    times = np.arange(0, cfg['orbit']['period_tu'], step)
    spread = np.array([setup['position_spread_km']/cfg['length_unit_km']]*3 +
                      [setup['velocity_spread_m_s']*.001*cfg['time_unit_s']/cfg['length_unit_km']]*3)
    streams = np.random.SeedSequence(setup['seed']).spawn(3)
    summary = {}
    for split, stream in zip(['training', 'validation', 'test'], streams):
        rng = np.random.default_rng(stream)
        count = setup[split+'_arcs']
        states = []; physical = []; measured = []; ideal = []; masks = []; elevations = []
        initial = np.asarray(cfg['orbit']['state']) + rng.normal(size=(count, 6))*spread
        for x0 in initial:
            trajectory = propagate(x0, times, cfg['mu'])
            obs = observations(times, trajectory, cfg, int(rng.integers(0, 2**32)))
            states.append(trajectory)
            physical.append([np.concatenate(to_mci(t, x, cfg)) for t, x in zip(times, trajectory)])
            measured.append(obs['measured_angles_rad']); ideal.append(obs['ideal_angles_rad'])
            masks.append(obs['sensor_visible']); elevations.append(obs['sensor_elevation_rad'])
        states = np.asarray(states)
        np.savez_compressed(directory/(split+'.npz'), times_tu=times,
            times_s=times*cfg['time_unit_s'], initial_states_normalized=initial,
            truth_rotating_normalized=states, truth_mci=physical,
            measured_angles_rad=measured, ideal_angles_rad=ideal,
            sensor_visible=masks, sensor_elevation_rad=elevations,
            transition_inputs=states[:, :-1], transition_targets=states[:, 1:])
        summary[split] = dict(arcs=count, samples_per_arc=len(times),
            transitions=count*(len(times)-1), visible_fraction=float(np.mean(np.any(masks, axis=2))))
    metadata = dict(summary=summary, config=cfg,
        axes='state arrays: arc,time,component; measurements: arc,time,sensor,angle',
        components=dict(truth_rotating_normalized='x,y,z,vx,vy,vz; barycentric rotating',
            truth_mci='x,y,z km; vx,vy,vz km/s; ax,ay,az km/s²; Moon-centered inertial',
            angles='alpha,delta rad; independent 1-sigma noise from config'),
        purpose='Local unforced dynamics identification from simulated truth states; measurements are separate noisy outputs.',
        splitting='Independent initial conditions and noise streams; never construct lag windows across arcs.',
        preprocessing='Fit scaling on training data only. Validation selects models; reserve test for final evaluation.',
        limitations='No thrust inputs, perturbing forces, or fitted model. Ideal angles include invisible geometry for diagnostics only; measured angles are NaN when unavailable.')
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return summary
