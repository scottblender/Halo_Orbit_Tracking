"""Run and summarize matched annual trials using the current LSO configuration."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from halo_tracking import lso_stationkeeping as sk

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('global', 'periodic')
TRIALS = 20
SEED = 20261122


def orbit_model():
    return sk.OrbitModel(*sk.physics.load_settings(ROOT))


def fit(output):
    orbit = orbit_model()
    settings = orbit.settings
    train = sk.make_experiment(orbit, settings['training_arcs'], settings['seed'])
    np.savez_compressed(output, global_gains=sk.fit_greybox(orbit, train)['gains'],
                        periodic_gains=sk.fit_periodic_greybox(
                            orbit, train, windows=settings['periodic_gain_windows'],
                            smoothing=settings['periodic_gain_smoothing'])['gains'])


def simulate(model, index, fits, output):
    orbit = orbit_model()
    settings = orbit.settings
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(TRIALS)[index-1])
    seed = int(rng.integers(2**32))
    sigma = np.repeat([settings['initial_position_sigma_km'], settings['initial_velocity_sigma_m_s']], 3)
    initial = rng.normal(size=6)*sigma
    with np.load(fits) as data:
        gains = data[model+'_gains']
    result = sk.stationkeeping_trial(
        orbit, gains, seed=seed, navigation='extended', initial_departure=initial,
        duration_days=settings['annual_horizon_days'], stop_on_departure_cutoff=False,
        planning_stride_steps=settings['annual_planning_stride_steps'],
    )
    summary = dict(result['summary'], model=model, trial=index, initial_departure=initial.tolist())
    day = result['day']
    values = np.linalg.norm(result['output'].reshape(-1, 2, 3), axis=2)
    if not summary['completed'] or not np.isfinite(values).all() or np.any(np.diff(day) < 0):
        raise ValueError(f'{model} trial {index} returned an incomplete or invalid propagation')
    np.savez_compressed(output, day=day, values=values, summary=json.dumps(summary))
    print(json.dumps(summary), flush=True)


def aggregate(directory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg, settings = sk.physics.load_settings(ROOT)
    limits = np.array([settings['position_limit_km'], settings['velocity_limit_m_s']])
    horizon = settings['annual_horizon_days']
    grid = None
    samples, rows, pair_id = {}, {}, {}
    for model in MODELS:
        samples[model], rows[model] = [], []
        for index in range(1, TRIALS+1):
            with np.load(directory/f'{model}-{index}.npz') as data:
                day, values = data['day'], data['values']
                row = json.loads(data['summary'].item())
            if (row['model'] != model or row['trial'] != index or not row['completed']
                    or day.ndim != 1 or not np.isfinite(day).all() or np.any(np.diff(day) < 0)
                    or day[0] != 0 or not np.isclose(day[-1], horizon, rtol=0, atol=1e-9)
                    or values.shape != (day.size, 2) or not np.isfinite(values).all()):
                raise ValueError(f'Invalid data for {model} trial {index}')
            if grid is None:
                grid = day
            if (day.shape != grid.shape or not np.array_equal(np.diff(day) == 0, np.diff(grid) == 0)
                    or np.max(np.abs(day-grid)) > 1e-5):
                raise ValueError(f'Unmatched sample layout for {model} trial {index}')
            identity = (row['seed'], row['initial_departure'])
            if index in pair_id and identity != pair_id[index]:
                raise ValueError(f'Unmatched initial state or noise seed for trial {index}')
            pair_id[index] = identity
            rows[model].append(row)
            samples[model].append(values)

    critical = t.ppf(.975, TRIALS-1)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(trials_per_model=TRIALS, days=horizon, sites=settings.get('lso_sensor_sites', cfg['sensors']), models={})
    fig, axes = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(11, 6.6))
    fig.subplots_adjust(left=.107, right=.98, top=.77, bottom=.14, wspace=.14, hspace=.30)
    fig.text(.105, .928, 'Annual station-keeping with revised LSO sites', fontsize=20, weight='bold', color='#172a44')
    fig.text(.105, .885, f'{TRIALS} matched trials per gain model with the same extended filter and planner', fontsize=11)
    upper = limits.copy()
    for col, model in enumerate(MODELS):
        values = np.stack(samples[model])
        mean = values.mean(axis=0)
        half = critical*values.std(axis=0, ddof=1)/np.sqrt(TRIALS)
        lo, hi = np.maximum(0, mean-half), mean+half
        dv = np.array([row['annual_delta_v_m_s'] for row in rows[model]])
        passed = (values <= limits).all(axis=(1,2))
        summary['models'][model] = dict(full_year_passes=int(passed.sum()),
            peaks_of_mean=mean.max(axis=0).tolist(), worst_trial_peaks=values.max(axis=(0,1)).tolist(),
            mean_inside_both_limits=bool((mean <= limits).all()),
            mean_commanded_delta_v_m_s=float(dv.mean()),
            delta_v_range_m_s=[float(dv.min()), float(dv.max())])
        axes[0,col].set_title(f'{model.title()} gains ({passed.sum()}/{TRIALS} full-year passes)', fontsize=11, weight='bold')
        upper = np.maximum(upper, hi.max(axis=0))
        for row, ax in enumerate(axes[:,col]):
            ax.fill_between(grid, lo[:,row], hi[:,row], color='#80AAD7', alpha=.48, linewidth=0,
                            label='Pointwise 95% CI for mean')
            ax.plot(grid, mean[:,row], color='#174E86', linewidth=1.05, label='Mean error')
            ax.axhline(limits[row], color='#B73536', ls='--', linewidth=1.1, label='Corridor limit')
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#DBE4ED', linewidth=.7)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(0, horizon)
        np.savetxt(output/f'{model}_mean_ci.csv', np.column_stack([grid, mean, lo, hi]),
                   fmt='%.12g', delimiter=',', comments='',
                   header='day,mean_position_km,mean_velocity_m_s,lower_position_km,lower_velocity_m_s,upper_position_km,upper_velocity_m_s')
    for row, label in enumerate(['Position error (km)', 'Velocity error (m/s)']):
        ax = axes[row,0]
        ax.set_ylabel(label)
        if upper[row] > 10*limits[row]:
            ax.set_yscale('symlog', linthresh=limits[row])
        ax.set_ylim(0, 1.12*upper[row])
    for ax in axes[-1]:
        ax.set_xlabel('Elapsed time (days)')
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper left', bbox_to_anchor=(.096, .866), ncol=3, frameon=False)
    fig.text(.105, .032, 'The band estimates uncertainty in the mean at each sampled time.', fontsize=9)
    fig.savefig(output/'stationkeeping_mc_95ci.png', dpi=190)
    plt.close(fig)
    delta = np.array([p['annual_delta_v_m_s']-g['annual_delta_v_m_s']
                      for g,p in zip(rows['global'],rows['periodic'])])
    width = critical*delta.std(ddof=1)/np.sqrt(TRIALS)
    summary['periodic_minus_global_delta_v_m_s'] = dict(mean=float(delta.mean()),
        confidence_interval_95=[float(delta.mean()-width),float(delta.mean()+width)])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    columns = ('model', 'trial', 'seed', 'all_evaluated_samples_inside', 'first_sampled_exit_day',
               'maximum_position_km', 'maximum_velocity_m_s', 'annual_delta_v_m_s', 'burns')
    with (output/'trials.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows([row[k] for k in columns] for model in MODELS for row in rows[model])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    fit_cmd = sub.add_parser('fit')
    fit_cmd.add_argument('output', type=Path)
    trial_cmd = sub.add_parser('trial')
    trial_cmd.add_argument('model', choices=MODELS)
    trial_cmd.add_argument('index', type=int, choices=range(1, TRIALS+1))
    trial_cmd.add_argument('fits', type=Path)
    trial_cmd.add_argument('output', type=Path)
    agg = sub.add_parser('aggregate')
    agg.add_argument('directory', type=Path)
    agg.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.action == 'fit':
        fit(args.output)
    elif args.action == 'trial':
        simulate(args.model, args.index, args.fits, args.output)
    else:
        aggregate(args.directory, args.output)
