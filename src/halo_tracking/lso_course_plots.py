"""Studio-style plots for the visibility-limited system identification study."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .lso_report_plots import BLUE, GREEN, RED, finish
from .stationkeeping_diagnostics import segment_stats, welch_bundle


ANGLE_NAMES = ('Right ascension', 'Declination')
COLORS = (BLUE, GREEN, RED)
FIGURE_STYLE = {'axes.labelsize': 10, 'axes.titlesize': 11, 'font.weight': 'normal',
                'axes.labelweight': 'normal', 'xtick.labelsize': 9, 'ytick.labelsize': 9}


def visible_runs(angles):
    edges = np.flatnonzero(np.diff(np.r_[False, np.isfinite(angles), False]))
    return list(zip(edges[::2], edges[1::2]))


def longest_bounds(angles):
    return max(visible_runs(angles), key=lambda bounds: bounds[1]-bounds[0], default=(0, 0))


@plt.rc_context(FIGURE_STYLE)
def ensemble_moments(orbit, offsets, split, out):
    day = np.arange(offsets.shape[1])*orbit.dt_s/86400
    fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
    for site, column in enumerate(axes.T):
        angles = offsets[..., 2*site:2*site+2]
        counts = np.isfinite(angles).sum(axis=0)
        means = np.divide(np.nansum(angles, axis=0), counts,
                          out=np.full(counts.shape, np.nan), where=counts > 0)
        variance = np.divide(np.nansum((angles-means)**2, axis=0), counts-1,
                             out=np.full(counts.shape, np.nan), where=counts > 1)
        paired = np.isfinite(angles).all(axis=-1)
        count = paired.sum(axis=0)
        pair_mean = np.divide(np.where(paired[..., None], angles, 0).sum(axis=0),
                              count[:, None], out=np.full(counts.shape, np.nan),
                              where=count[:, None] > 0)
        product = np.prod(angles-pair_mean, axis=-1)
        covariance = np.divide(np.where(paired, product, 0).sum(axis=0), count-1,
                               out=np.full(len(day), np.nan), where=count > 1)
        for c, name in enumerate(ANGLE_NAMES):
            column[0].plot(day, means[:, c], color=COLORS[c], label=name)
            column[1].plot(day, np.sqrt(variance[:, c]), color=COLORS[c])
        column[2].plot(day, covariance, color=BLUE)
        column[2].axhline(0, color='grey', lw=.7)
        column[3].step(day, count, where='mid', color='grey')
        column[3].set(xlabel='Elapsed time [day]', ylim=(-.2, len(offsets)+.2))
        column[0].set_title(f'LSO {site+1}')
    axes[0, 0].legend(fontsize=8)
    for ax, label in zip(axes[:, 0], ('Mean μ[t] [arcsec]', 'Std. dev. σ[t] [arcsec]',
                                       'Cov(α, δ) [arcsec²]', 'Visible trajectories')):
        ax.set_ylabel(label)
    finish(fig, out, f'{split}_angle_moments',
           f'Estimate moments at each orbit epoch across {len(offsets)} independent {split} trajectories.',
           'The traces show the mean, scatter and paired-angle covariance changing over visible phases. Standard deviation and covariance require at least two visible trajectories.')


@plt.rc_context(FIGURE_STYLE)
def angle_segments(orbit, offsets, rows, seeds, split, out):
    hours = orbit.dt_s/3600
    colors = plt.cm.viridis(np.linspace(.08, .87, 4))
    for angle, name in enumerate(ANGLE_NAMES):
        fig, axes = plt.subplots(4, 3, figsize=(12, 9))
        for site, column in enumerate(axes.T):
            c = 2*site+angle
            row = rows[c]
            arc = row['arc']
            start, stop = longest_bounds(offsets[arc, :, c])
            column[0].set_title(f'LSO {site+1} · Seed {seeds[arc]}')
            if row['adf_p'] is None:
                column[0].text(.5, .5, 'Too few consecutive samples',
                               transform=column[0].transAxes, ha='center')
                continue
            stats = segment_stats(offsets[arc, start:stop, c])
            centres = (start+(np.arange(4)+.5)*stats['L']-.5)*hours/24
            for ax, moment, error in zip(column[:2], ('mu', 'var'), ('se_mu', 'se_var')):
                estimates = stats[moment]
                level, band = estimates.mean(), 1.96*stats[error]
                marked = abs(estimates-level) > band
                ax.axhspan(level-band, level+band, color='grey', alpha=.16)
                ax.axhline(level, color='grey', lw=.8)
                ax.plot(centres, estimates, 'o-', color=BLUE, ms=4)
                ax.plot(centres[marked], estimates[marked], 'o', mfc='none', mec=RED, ms=8)
            lag = np.arange(stats['M']+1)*hours
            for i, (cov, color) in enumerate(zip(stats['G'], colors), 1):
                column[2].plot(lag, cov, 'o-', color=color, ms=3, label=f'Segment {i}')
                if cov[0] > 0:
                    column[3].plot(lag[1:], cov[1:]/cov[0], 'o-', color=color, ms=3)
            column[1].set_xlabel('Segment centre [day]')
            column[3].set_xlabel('Lag [hour]')
            for ax in column[2:]:
                ax.axhline(0, color='grey', lw=.7)
        for ax, label in zip(axes[:, 0], ('Mean [arcsec]', 'Variance [arcsec²]',
                                           'Autocovariance [arcsec²]', 'ACF')):
            ax.set_ylabel(label)
        if axes[2, 0].get_legend_handles_labels()[0]:
            axes[2, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(name)
        finish(fig, out, f'{split}_angle_segments_{angle+1}',
               'Compare the four consecutive subsegments of each longest visible run, following Studio 6.',
               'Red rings mark values outside the approximate correlation-aware 95% bands. The autocovariance and normalized ACF show how lag structure changes between segments.')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for site, ax in enumerate(axes):
        arc = rows[2*site]['arc']
        pair = offsets[arc, :, 2*site:2*site+2]
        start, stop = longest_bounds(np.where(np.isfinite(pair).all(axis=1), pair[:, 0], np.nan))
        length = (stop-start)//4
        ax.set(title=f'LSO {site+1} · Seed {seeds[arc]}', xlabel='Lag [hour]')
        if length < 4:
            ax.text(.5, .5, 'Too few paired samples', transform=ax.transAxes, ha='center')
            continue
        lags = np.arange(-(length//4), length//4+1)
        for i, color in enumerate(colors):
            x, y = pair[start+i*length:start+(i+1)*length].T
            x, y = x-x.mean(), y-y.mean()
            scale = np.linalg.norm(x)*np.linalg.norm(y)
            if scale > 0:
                ccf = signal.correlate(y, x)[length-1+lags]/scale
                ax.plot(lags*hours, ccf, 'o-', color=color, ms=3, label=f'Segment {i+1}')
        ax.axhline(0, color='grey', lw=.7)
        ax.axvline(0, color='grey', lw=.7, ls=':')
    axes[0].set_ylabel('Inter-angle CCF')
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7, ncol=2)
    finish(fig, out, f'{split}_angle_ccf',
           'Compare right ascension and declination cross-correlation across four paired segments.',
           'Positive lags put right ascension first. Segment-to-segment changes indicate changes in the joint lag structure.')


def angle_welch(commands, offsets, fs, length):
    """Fit matched command–angle spectra without spanning missing observations."""
    results = []
    for channel in range(offsets.shape[-1]):
        u, y = [], []
        for arc, angles in enumerate(offsets[:, :-1, channel]):
            for start, stop in visible_runs(angles):
                for k in range(start, stop-length+1, length):
                    u.append(commands[arc, k:k+length])
                    y.append(angles[k:k+length, None])
        results.append(welch_bundle(np.asarray(u), np.asarray(y), fs, length, overlap=0.)
                       if u else None)
    return results


@plt.rc_context(FIGURE_STYLE)
def measured_frf(spectra, length, fs, threshold, out):
    for c, angle in enumerate(ANGLE_NAMES):
        fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
        for site, column in enumerate(axes.T):
            spec = spectra[2*site+c]
            column[0].set_title(f'LSO {site+1}')
            if spec is None:
                column[0].text(.5, .5, 'No visible windows', transform=column[0].transAxes, ha='center')
                continue
            column[0].set_title(f'LSO {site+1} · {spec["segments"]} windows')
            f = spec['f'][1:]*86400
            column[0].semilogy(f, spec['syy'][1:, 0]/86400, '.-', color=BLUE)
            for j, color in enumerate(COLORS):
                response = spec['H'][1:, 0, j]
                column[1].semilogy(f, abs(response), '.-', color=color, label='XYZ'[j])
                column[2].plot(f, np.angle(response, deg=True), '.', color=color)
                column[3].plot(f, spec['coherence'][1:, 0, j], '.-', color=color)
            column[3].axhline(threshold, color='grey', ls='--', lw=.8)
        for column in axes.T:
            column[2].set_ylim(-190, 190)
            column[3].set(xlabel='Frequency [cycles/day]', ylim=(0, 1.04))
        if axes[1, 0].get_legend_handles_labels()[0]:
            axes[1, 0].legend(title='Burn axis', fontsize=7, ncol=3)
        for ax, label in zip(axes[:, 0], ('Angle PSD [arcsec²/(cycle/day)]',
                                           'Gain [arcsec/(m/s)]', 'Phase [deg]', 'Pairwise coherence')):
            ax.set_ylabel(label)
        fig.suptitle(angle)
        finish(fig, out, f'welch_measured_angles_{c+1}',
               'Estimate the measured-angle PSD and the three-input FRF from matched, disjoint Hann windows.',
               f'The bin spacing is {86400*fs/length:.3g} cycles/day. Each site samples different phases. Phase is wrapped and the dashed line marks the pairwise coherence screen.')


@plt.rc_context(FIGURE_STYLE)
def residual_checks(reports, split, alpha, out):
    channels = next(iter(reports.values()))['acf'].shape[1]
    fig, axes = plt.subplots(channels, len(reports), figsize=(10, 10),
                             sharex=True, sharey=True, squeeze=False)
    results = []
    for j, (name, report) in enumerate(reports.items()):
        axes[0, j].set_title(name)
        lag = np.arange(1, report['lags']+1)
        for c, ax in enumerate(axes[:, j]):
            acf = report['acf'][:, c]
            count = report['acf_pairs'][:, c]
            band = np.divide(1.96, np.sqrt(count), out=np.full(len(lag), np.nan),
                             where=np.isfinite(acf))
            ax.fill_between(lag, -band, band, color='grey', alpha=.17)
            ax.vlines(lag, 0, acf, color=BLUE, lw=1.4)
            ax.axhline(0, color='grey', lw=.7)
            if j == 0:
                ax.set_ylabel(f'ACF coord. {c+1}')
        axes[-1, j].set_xlabel('Innovation lag [samples]')
        rejects = np.count_nonzero(report['ljung_box_holm_p'] < alpha)
        results.append(f'{name}: {rejects}/{report["eligible_whiteness_tests"]} rejected')
    finish(fig, out, f'{split}_innovation_acf',
           'Check H by plotting standardized innovation ACF with approximate pointwise white-noise bounds.',
           'The formal decision uses Ljung–Box on contiguous runs with Holm adjustment. ' + '. '.join(results) + '.')

    fig, axes = plt.subplots(channels, len(reports), figsize=(10, 10),
                             sharex=True, sharey=True, squeeze=False)
    for j, (name, report) in enumerate(reports.items()):
        axes[0, j].set_title(name)
        delay = np.arange(1, report['lags']+2)
        for c, ax in enumerate(axes[:, j]):
            for k, color in enumerate(COLORS):
                ax.plot(delay, report['ccf'][:, c, k], '.-', color=color, label='XYZ'[k])
            ax.axhline(0, color='grey', lw=.7)
            if j == 0:
                ax.set_ylabel(f'CCF coord. {c+1}')
        axes[-1, j].set_xlabel('Command-to-innovation delay [samples]')
    axes[0, 0].legend(title='Burn axis', fontsize=7, ncol=3)
    finish(fig, out, f'{split}_command_innovation_ccf',
           'Check G by plotting each commanded axis against the standardized innovations at its actual delay.',
           'The first lag pairs u[k] with ε[k+1]. The accompanying maximum-correlation test uses randomized commands and accounts for the full set of plotted correlations.')
