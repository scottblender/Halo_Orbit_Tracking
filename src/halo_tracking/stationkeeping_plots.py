"""Readable figures for the two course notebooks; all numerical work elsewhere."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .stationkeeping_diagnostics import segment_stats

BLUE, GREEN, RED = '#1414B4', '#0B9A0B', '#C00000'
NAMES = ['r x', 'r y', 'r z', 'v x', 'v y', 'v z']


def style():
    plt.rcParams.update({'font.size':15, 'font.weight':'bold',
                         'axes.labelsize':16, 'axes.labelweight':'bold',
                         'axes.titlesize':17, 'axes.titleweight':'bold',
                         'xtick.labelsize':13, 'ytick.labelsize':13,
                         'legend.fontsize':12, 'figure.dpi':110,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'lines.linewidth':1.8, 'grid.alpha':.22,
                         'savefig.facecolor':'white'})


def finish(fig, output, name):
    fig.tight_layout()
    Path(output).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output)/(name+'.png'), dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def input_output(orbit, rec, output):
    day = np.arange(orbit.n+1)*orbit.dt_s/86400
    fig, axs = plt.subplots(3, 3, figsize=(19, 12), sharex=True)
    for c in range(3):
        axs[0, c].step(day[:-1], rec.u[0, :, c], where='post', color=BLUE)
        axs[0, c].set_title('xyz'[c]+' direction')
        axs[0, c].set_ylabel('Command [m/s]')
        axs[1, c].plot(day, rec.y[0, :, c], color=BLUE)
        axs[1, c].set_ylabel('Position error [km]')
        axs[2, c].plot(day, rec.y[0, :, c+3], color=BLUE)
        axs[2, c].set_ylabel('Velocity error [m/s]')
        axs[2, c].set_xlabel('Elapsed time [day]')
    for ax in axs.flat:
        ax.grid()
    finish(fig, output, '01_input_output_xyz')


def spectra(spec, orbit, output, suffix=''):
    f = spec['f']*86400  # convert Hz to cycles/day; density transformed too
    fig, axs = plt.subplots(3, 3, figsize=(19, 12), sharex=True)
    lo, hi = spec['psd_band_factor']
    for c in range(3):
        p = spec['suu'][:, c]/86400
        axs[0, c].semilogy(f[1:], p[1:], color=BLUE)
        axs[0, c].fill_between(f[1:], lo*p[1:], hi*p[1:], color=BLUE, alpha=.12)
        axs[0, c].set_title('xyz'[c]+' direction')
        axs[0, c].set_ylabel('Input PSD\n[(m/s)²/(cycle/day)]')
        for row, channel, unit in [(1,c,'km²'), (2,c+3,'(m/s)²')]:
            axs[row, c].semilogy(f[1:], spec['syy'][1:, channel]/86400, color=BLUE)
            axs[row, c].set_ylabel('Output PSD\n['+unit+'/(cycle/day)]')
        axs[2, c].set_xlabel('Frequency [cycle/day]')
    for ax in axs.flat:
        ax.grid()
    finish(fig, output, '02_welch_psd'+suffix)


def frf(spec, output, model_spec=None):
    f = spec['f']*86400
    for block, offset, units in [('position',0,'km/(m/s)'), ('velocity',3,'(m/s)/(m/s)')]:
        fig, axs = plt.subplots(3, 3, figsize=(19, 12), sharex=True)
        for r in range(3):
            for c in range(3):
                ax = axs[r, c]
                h = abs(spec['H'][:, r+offset, c])
                ax.semilogy(f[1:], h[1:], color=BLUE, label='Empirical MIMO H1')
                coh = spec['coherence'][:, r+offset, c]
                ax.scatter(f[1:][coh[1:]<.8], h[1:][coh[1:]<.8], color=RED, s=12,
                           label='Pairwise coherence < 0.8')
                if model_spec is not None:
                    ax.semilogy(f[1:], abs(model_spec['H'][1:, r+offset,c]), '--',
                                color=GREEN, label='Grey-box free simulation')
                ax.set_title('Δv '+ 'xyz'[c]+' → '+block+' '+'xyz'[r])
                if c == 0:
                    ax.set_ylabel('Gain ['+units+']')
                if r == 2:
                    ax.set_xlabel('Frequency [cycle/day]')
                ax.grid()
        axs[0, 0].legend(fontsize=9)
        finish(fig, output, '03_frf_'+block+('_comparison' if model_spec else ''))
    if model_spec is None:
        fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
        for i, ax in enumerate(axs.flat):
            for c in range(3):
                ax.plot(f[1:], spec['coherence'][1:, i,c], label='Δv '+'xyz'[c])
            ax.axhline(.8, color=RED, ls=':')
            ax.set(title=NAMES[i], ylim=(0,1.05), xlabel='Frequency [cycle/day]', ylabel='Pairwise coherence')
        axs[0,0].legend()
        finish(fig, output, '04_coherence')
        fig, axs = plt.subplots(2,3,figsize=(18,8),sharex=True)
        for i,ax in enumerate(axs.flat):
            for c in range(3):
                ax.plot(f[1:], np.angle(spec['H'][1:, i,c], deg=True), label='Δv '+'xyz'[c])
            ax.set(title=NAMES[i], ylabel='Wrapped phase [deg]', xlabel='Frequency [cycle/day]')
        axs[0,0].legend()
        finish(fig,output,'05_frf_phase')
    else:
        fig, axs = plt.subplots(2,3,figsize=(18,8),sharex=True)
        for i,ax in enumerate(axs.flat):
            c=i%3
            ax.plot(f[1:],np.angle(spec['H'][1:,i,c],deg=True),color=BLUE,label='Empirical')
            ax.plot(f[1:],np.angle(model_spec['H'][1:,i,c],deg=True),'--',color=GREEN,label='Grey box')
            ax.set(title='Δv '+'xyz'[c]+' → '+NAMES[i],ylabel='Wrapped phase [deg]',xlabel='Frequency [cycle/day]')
        axs[0,0].legend()
        finish(fig,output,'05b_frf_phase_comparison')


def stationarity_panels(rec, orbit, output, prefix='06_stationarity'):
    fig, axs = plt.subplots(2, 6, figsize=(25, 9))
    for c in range(6):
        st = segment_stats(rec.y[0, :, c])
        t = (np.arange(4)+.5)*st['L']*orbit.dt_s/86400
        for row, key, se, lab in [(0,'mu','se_mu','Mean'), (1,'var','se_var','Variance')]:
            ax = axs[row,c]
            values = st[key]
            ax.axhspan(values.mean()-1.96*st[se], values.mean()+1.96*st[se], color='grey',alpha=.18)
            ax.plot(t, values, 'o-', color=BLUE)
            ax.set(xlabel='Day', ylabel=lab, title=NAMES[c])
    finish(fig, output, prefix)


def structure(orbit, rec, diagnostics, ccf, spec, output):
    fig, axs = plt.subplots(2,3,figsize=(18,9),sharex=True)
    for i,ax in enumerate(axs.flat):
        for c in range(3):
            ax.plot(np.arange(len(ccf))*orbit.dt_s/3600, ccf[:,i,c],label='Δv '+'xyz'[c])
        ax.axhline(0,color='grey',lw=.7)
        ax.set(title=NAMES[i],xlabel='Lag [hour]',ylabel='Input–output correlation')
    axs[0,0].legend()
    finish(fig,output,'07_lag_structure')
    fig, axs = plt.subplots(1,3,figsize=(19,5))
    normalized = rec.u.reshape(-1,3)/rec.u.reshape(-1,3).std(axis=0)
    sv = np.linalg.svd(normalized,compute_uv=False)
    axs[0].bar(np.arange(1,4),sv/sv[0],color=BLUE)
    axs[0].set(xlabel='Input singular value',ylabel='Relative magnitude',xticks=[1,2,3])
    sens = diagnostics['impulse_sensitivity']
    for c in range(3):
        axs[1].plot(diagnostics['phase_deg'], np.linalg.norm(sens[:,:,c],axis=1),label='Δv '+'xyz'[c])
    axs[1].set(xlabel='Orbit phase [deg]',ylabel='Scaled horizon response / (m/s)')
    axs[1].legend()
    axs[2].loglog(diagnostics['amplitudes_km'],diagnostics['linearity_relative_error'],'o-',alpha=.65)
    axs[2].set(xlabel='Perturbation scale [km]',ylabel='STM / nonlinear relative error')
    for ax in axs:
        ax.grid()
    finish(fig,output,'08_excitation_phase_linearity')
    fig, axs = plt.subplots(1,2,figsize=(14,6))
    coupling = np.sqrt(np.mean(sens**2,axis=0))
    im=axs[0].imshow(coupling,aspect='auto',cmap='viridis')
    axs[0].set(xticks=range(3),xticklabels=['Δv x','Δv y','Δv z'],
               yticks=range(6),yticklabels=NAMES,title='Phase-averaged coupling')
    fig.colorbar(im,ax=axs[0],label='RMS scaled sensitivity / (m/s)')
    angles=np.linspace(0,2*np.pi,300)
    axs[1].plot(np.cos(angles),np.sin(angles),':',color='grey')
    eig=diagnostics['multipliers']
    axs[1].scatter(eig.real,eig.imag,s=65,color=BLUE)
    axs[1].set(xlabel='Real multiplier',ylabel='Imaginary multiplier',title='One-orbit Floquet multipliers',aspect='equal')
    finish(fig,output,'09_coupling_stability')


def geometry(orbit, days, departure, output):
    times=np.linspace(0,orbit.T,600,endpoint=False)
    xyz=(orbit.reference(times)[:3].T-np.array([1-orbit.cfg['mu'],0,0]))*orbit.cfg['length_unit_km']
    fig=plt.figure(figsize=(17,7))
    ax=fig.add_subplot(121,projection='3d')
    ax.plot(*xyz.T,color=BLUE,label='Nominal NRHO')
    ax.scatter(0,0,0,color='grey',s=65,label='Moon')
    ax.set(xlabel='x [km]',ylabel='y [km]',zlabel='z [km]')
    ax.legend()
    ax=fig.add_subplot(222)
    ax.semilogy(days,np.linalg.norm(departure[:,:3],axis=1),color=BLUE)
    ax.axhline(orbit.settings['position_limit_km'],color=RED,ls='--')
    ax.set(xlabel='Day',ylabel='Position error [km]')
    ax=fig.add_subplot(224)
    ax.semilogy(days,np.linalg.norm(departure[:,3:],axis=1),color=BLUE)
    ax.axhline(orbit.settings['velocity_limit_m_s'],color=RED,ls='--')
    ax.set(xlabel='Day',ylabel='Velocity error [m/s]')
    finish(fig,output,'10_geometry_uncontrolled')


def residuals(report, rec, orbit, output):
    fig,axs=plt.subplots(2,3,figsize=(18,9),sharex=True)
    for c,ax in enumerate(axs.flat):
        r=report['acf'][:,c]
        for a in r:
            ax.plot(np.arange(1,len(a)+1),a,alpha=.7)
        bound=1.96/np.sqrt(rec.u.shape[1])
        ax.axhspan(-bound,bound,color='grey',alpha=.18)
        ax.set(title=NAMES[c],xlabel='Lag [sample]',ylabel='Innovation ACF')
    finish(fig,output,'11_innovation_acf')
    fig,axs=plt.subplots(2,3,figsize=(18,9),sharex=True)
    for c,ax in enumerate(axs.flat):
        for j in range(3):
            ax.plot(np.arange(report['lags']+1),report['ccf'][:,c,j],label='Δv '+'xyz'[j])
        bound=1.96/np.sqrt(report['ccf_pairs'])
        ax.fill_between(np.arange(len(bound)),-bound,bound,color='grey',alpha=.18)
        ax.set(title=NAMES[c],xlabel='Earlier-command lag [sample]',ylabel='Innovation–input CCF')
    axs[0,0].legend()
    finish(fig,output,'12_innovation_input_ccf')


def prediction(orbit, rec, pred, free, output):
    day=np.arange(1,orbit.n+1)*orbit.dt_s/86400
    fig,axs=plt.subplots(2,3,figsize=(18,9),sharex=True)
    for i,ax in enumerate(axs.flat):
        ax.plot(day,rec.y[0,1:,i],color=BLUE,label='Held-out observation')
        ax.plot(day,pred['predicted'][0,:,i],'--',color=GREEN,label='One-step prediction')
        ax.plot(day,free['predicted'][0,:,i],':',color=RED,label='Free simulation')
        ax.set(title=NAMES[i],xlabel='Day',ylabel='km' if i<3 else 'm/s')
    axs[0,0].legend()
    finish(fig,output,'13_heldout_prediction')


def controlled(orbit, trial, no_burn_day, no_burn_output, output):
    fig,axs=plt.subplots(3,1,figsize=(13,13),sharex=True)
    for ax,sl,limit,label in [(axs[0],slice(0,3),orbit.scale[0],'Position error [km]'),
                               (axs[1],slice(3,6),orbit.scale[3],'Velocity error [m/s]')]:
        ax.semilogy(no_burn_day,np.linalg.norm(no_burn_output[:,sl],axis=1),color=RED,label='No burns')
        ax.semilogy(trial['day'],np.linalg.norm(trial['output'][:,sl],axis=1),color=BLUE,label='Grey-box guidance')
        ax.axhline(limit,color='black',ls='--',label='Required corridor')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid()
    if len(trial['commands']):
        axs[2].stem(trial['burn_day'],np.linalg.norm(trial['commands'],axis=1),basefmt=' ')
    axs[2].set(xlabel='Elapsed time [day]',ylabel='Commanded impulse [m/s]')
    finish(fig,output,'14_stationkeeping_corridor')
    fig,axs=plt.subplots(1,2,figsize=(15,5))
    if len(trial['commands']):
        axs[0].scatter(trial['phase_deg'],np.linalg.norm(trial['commands'],axis=1),color=BLUE)
        for c in range(3):
            axs[1].plot(trial['burn_day'],trial['commands'][:,c],'o-',label='xyz'[c])
        axs[1].legend()
    axs[0].set(xlabel='Selected burn phase [deg]',ylabel='Command magnitude [m/s]',xlim=(0,360))
    axs[1].set(xlabel='Day',ylabel='Command component [m/s]')
    finish(fig,output,'15_magnitude_phase')
