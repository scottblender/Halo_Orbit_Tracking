"""Concise notebook figures, each with an evidence-based two-sentence caption."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from IPython.display import display, Markdown
from .measurements import ARCSEC
from .stationkeeping_plots import style
from .stationkeeping_diagnostics import segment_stats
from .surface_sensor import site_rotating

BLUE, GREEN, RED = '#1414B4', '#0B9A0B', '#C00000'


def finish(fig,out,name,purpose,result):
    fig.tight_layout(pad=1.6)
    Path(out).mkdir(parents=True,exist_ok=True)
    fig.savefig(Path(out)/(name+'.png'),dpi=150,bbox_inches='tight')
    plt.show(); plt.close(fig)
    display(Markdown('**Purpose:** '+purpose+'\n\n**Result:** '+result))


def diagram(orbit,out):
    fig,ax=plt.subplots(figsize=(16,7))
    ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
    nodes=[(2,3.6,'Impulse actuator\ncommand × fitted gain'),
           (6,3.6,'Nonlinear NRHO dynamics\nEarth–Moon gravity'),
           (10,3.6,'Three lunar observatories\nline-of-sight geometry'),
           (10,1.3,'Angular observations\nnoise + visibility gaps'),
           (6,1.3,'STM state estimator\nsubtract nominal orbit'),
           (2,1.3,'Station-keeping planner\nimpulse magnitude + phase')]
    for x,y,text in nodes:
        ax.add_patch(FancyBboxPatch((x-1.65,y-.55),3.3,1.1,boxstyle='round,pad=0.08',
                                    facecolor='#eef2ff',edgecolor=BLUE,lw=1.6))
        ax.text(x,y,text,ha='center',va='center',fontsize=13,fontweight='bold')
    for start,end in [((3.75,3.6),(4.25,3.6)),((7.75,3.6),(8.25,3.6)),
                      ((10,2.95),(10,1.95)),((8.25,1.3),(7.75,1.3)),
                      ((4.25,1.3),(3.75,1.3)),((2,1.95),(2,2.95))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=2,color=BLUE))
    ax.text(6,.12,'Identification fits gains using raw angular innovations. Truth is reserved for simulation and evaluation.',
            ha='center',fontsize=12)
    finish(fig,out,'00_lso_feedback_diagram',
           'This diagram identifies the measurement and feedback paths used to keep the NRHO near its nominal orbit.',
           f'The {len(orbit.sites)} LSOs provide angles rather than direct position or velocity measurements, so visibility and state estimation are part of the station-keeping problem.')


def inputs_outputs(orbit,rec,out):
    day=np.arange(orbit.n+1)*orbit.dt_s/86400
    fig,axs=plt.subplots(3,3,figsize=(19,11),sharex=True)
    for c in range(3):
        axs[0,c].stem(day[:-1],rec.u[0,:,c],basefmt=' ',markerfmt=' ')
        axs[0,c].set(title='xyz'[c]+' direction',ylabel='Impulse command [m/s]')
        for row,index,label in [(1,c,'Position departure [km]'),(2,c+3,'Velocity departure [m/s]')]:
            axs[row,c].plot(day,rec.y[0,:,index],color=BLUE,label='LSO-derived estimate')
            axs[row,c].plot(day,rec.truth[0,:,index],':',color='grey',label='Truth: evaluation only')
            axs[row,c].set_ylabel(label)
        axs[2,c].set_xlabel('Elapsed time [day]')
    axs[1,0].legend(fontsize=10)
    for ax in axs.flat:ax.grid(alpha=.2)
    err=rec.y-rec.truth
    rmse=np.sqrt(np.mean(np.sum(err[...,:3]**2,axis=-1)))
    finish(fig,out,'01_input_output_xyz',
           'These plots compare the three commanded impulses with the six departures reconstructed from LSO angles.',
           f'The training navigation position RMSE is {rmse:.3g} km, showing the measurement-to-state uncertainty that the identification workflow must retain.')


def lso_measurements(orbit,rec,out):
    day=np.arange(orbit.n+1)*orbit.dt_s/86400
    nominal=np.asarray([orbit.observation_terms(k)[0] for k in range(orbit.n+1)])
    from .measurements import wrap
    delta=rec.angles[0]-nominal
    delta[...,0]=wrap(delta[...,0])
    fig,axs=plt.subplots(3,1,figsize=(14,10),sharex=True)
    for j,site in enumerate(orbit.sites):
        label=site['site'].get('name',f'LSO {j+1}')
        for c in range(2):
            axs[c].plot(day,delta[:,j,c]/ARCSEC,'.-',ms=3,label=label)
    axs[0].set_ylabel('α departure [arcsec]');axs[1].set_ylabel('δ departure [arcsec]')
    axs[0].legend(ncol=3,fontsize=11)
    axs[2].step(day,rec.visible[0].sum(axis=1),where='mid',color=BLUE)
    axs[2].set(xlabel='Elapsed time [day]',ylabel='Visible LSOs',yticks=range(len(orbit.sites)+1))
    coverage=np.mean(np.any(rec.visible,axis=-1))
    finish(fig,out,'02_raw_lso_angles_visibility',
           'The angular-departure plots show the actual noisy LSO data relative to the nominal line of sight, with missing samples left blank.',
           f'At least one LSO is visible at {100*coverage:.1f}% of training epochs, '+
           ('giving full network coverage in this case even though individual LSO channels are masked.' if coverage==1 else
            'so the state estimator must coast through epochs without a visible observatory.'))


def spectra(orbit,spec,out):
    f=spec['f']*86400
    fig,axs=plt.subplots(2,3,figsize=(18,8),sharex=True)
    for c in range(3):
        p=spec['suu'][:,c]/86400
        axs[0,c].semilogy(f[1:],p[1:],color=BLUE)
        lo,hi=spec['psd_band_factor']
        axs[0,c].fill_between(f[1:],lo*p[1:],hi*p[1:],alpha=.14,color=BLUE)
        axs[0,c].set(title='xyz'[c]+' direction',ylabel='Input PSD\n[(m/s)²/(cycle/day)]')
        axs[1,c].semilogy(f[1:],spec['syy'][1:,c]/86400,color=BLUE)
        axs[1,c].set(xlabel='Frequency [cycle/day]',ylabel='Estimated position PSD\n[km²/(cycle/day)]')
    finish(fig,out,'03_welch_psd',
           'Welch spectra compare input excitation with the frequency content of the LSO-derived position estimates.',
           f'The chosen window gives {f[1]-f[0]:.3g} cycle/day resolution and {spec["segments"]} windows, so these curves summarize a filtered finite record rather than establish a stationary plant spectrum.')


def frf(spec,out):
    f=spec['f']*86400
    fig,axs=plt.subplots(3,3,figsize=(18,11),sharex=True)
    for c in range(3):
        h=spec['H'][:,c,c];coh=spec['coherence'][:,c,c]
        axs[0,c].semilogy(f[1:],abs(h[1:]),color=BLUE)
        axs[0,c].set(title='Δv '+'xyz'[c]+' → position '+'xyz'[c],ylabel='Gain [km/(m/s)]')
        axs[1,c].plot(f[1:],np.angle(h[1:],deg=True),color=BLUE)
        axs[1,c].set_ylabel('Wrapped phase [deg]')
        axs[2,c].plot(f[1:],coh[1:],color=BLUE)
        axs[2,c].axhline(.8,color=RED,ls=':')
        axs[2,c].set(xlabel='Frequency [cycle/day]',ylabel='Pairwise coherence',ylim=(0,1.05))
    fraction=np.mean(np.stack([spec['coherence'][1:,c,c] for c in range(3)])>=.8)
    finish(fig,out,'04_frf_phase_coherence',
           'The displayed diagonal channels of the full MIMO estimate show response gain, phase and input–output coherence.',
           f'Only {100*fraction:.1f}% of these bins meet the 0.8 coherence screen, and the response includes LSO estimation dynamics, so a global LTI plant model is not justified by this plot alone.')


def stationarity(orbit,rec,rows,out):
    fig,axs=plt.subplots(2,3,figsize=(17,8))
    for c in range(3):
        st=segment_stats(rec.y[0,:,c]);t=(np.arange(4)+.5)*st['L']*orbit.dt_s/86400
        for r,key,se,label in [(0,'mu','se_mu','Mean [km]'),(1,'var','se_var','Variance [km²]')]:
            ax=axs[r,c];v=st[key]
            ax.axhspan(v.mean()-1.96*st[se],v.mean()+1.96*st[se],color='grey',alpha=.18)
            ax.plot(t,v,'o-',color=BLUE)
            ax.set(title='Estimated position '+'xyz'[c],ylabel=label,xlabel='Segment centre [day]')
    flags=sum(r['mean_flags']+r['variance_flags'] for r in rows)
    finish(fig,out,'05_stationarity',
           'Segment means and variances check whether the LSO-derived position record maintains comparable statistics over an orbit.',
           f'The six state channels produce {flags} mean/variance flags across their segment checks, so stationarity must be assessed with the accompanying ADF/KPSS results and visibility history rather than assumed.')


def structure(d,out):
    fig,axs=plt.subplots(1,3,figsize=(19,6))
    sens=d['impulse_sensitivity']
    for c in range(3):
        axs[0].plot(d['phase_deg'],np.linalg.norm(sens[:,:,c],axis=1),label='Δv '+'xyz'[c])
    axs[0].set(xlabel='Orbit phase [deg]',ylabel='Scaled horizon response / (m/s)');axs[0].legend()
    axs[1].loglog(d['amplitudes_km'],d['linearity_relative_error'],'o-',alpha=.65)
    axs[1].set(xlabel='Perturbation scale [km]',ylabel='STM / nonlinear relative error')
    axs[2].semilogy(np.arange(1,7),abs(d['multipliers']),'o',color=BLUE,ms=9)
    axs[2].axhline(1,color=RED,ls='--')
    axs[2].set(xlabel='Floquet multiplier index',ylabel='One-orbit multiplier magnitude',xticks=range(1,7))
    response=np.linalg.norm(sens,axis=(1,2));ratio=response.max()/max(response.min(),1e-15)
    radius=abs(d['multipliers']).max()
    finish(fig,out,'06_phase_linearity_stability',
           'These diagnostics separate phase-dependent sensitivity, finite-amplitude linearization error and one-orbit perturbation growth.',
           f'Horizon sensitivity varies by a factor of {ratio:.2g} over phase and the largest multiplier is {radius:.3g}, supporting a time-varying local model'+(' with an unstable mode.' if radius>1.001 else ' whose stability needs further resolution.'))


def geometry_growth(orbit,day,y,summary,out):
    # Orthographic projections avoid the overlapping 3-D labels of the old plot.
    xyz=(orbit.reference(np.linspace(0,orbit.T,800,endpoint=False))[:3].T-
         np.array([1-orbit.cfg['mu'],0,0]))*orbit.cfg['length_unit_km']/1000
    fig,axs=plt.subplots(2,2,figsize=(17,11))
    for ax,c in [(axs[0,0],1),(axs[1,0],2)]:
        ax.plot(xyz[:,0],xyz[:,c],color=BLUE,label='Nominal NRHO')
        ax.add_patch(Circle((0,0),orbit.cfg['site']['radius_km']/1000,color='grey',alpha=.6,label='Moon'))
        for j,site in enumerate(orbit.sites):
            q=site_rotating(site)*orbit.cfg['length_unit_km']/1000
            ax.scatter(q[0],q[c],s=25,label=f'LSO {j+1}')
        ax.set(xlabel='Moon-centred x [10³ km]',ylabel='Moon-centred '+'xyz'[c]+' [10³ km]',aspect='equal')
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=10,ncol=2)
    first=[]
    for ax,sl,limit,label in [(axs[0,1],slice(0,3),10.,'Position departure [km]'),
                              (axs[1,1],slice(3,6),.1,'Velocity departure [m/s]')]:
        value=np.linalg.norm(y[:,sl],axis=1)
        ax.semilogy(day,value,color=BLUE)
        ax.axhline(limit,color=RED,ls='--',label='Required limit')
        use=np.flatnonzero(value>limit)
        first.append(float(day[use[0]]) if len(use) else None)
        if len(use):ax.axvline(day[use[0]],color='grey',ls=':',lw=1)
        ax.set(xlabel='Elapsed time [day]',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=11)
    peakp=np.linalg.norm(y[:,:3],axis=1).max();peakv=np.linalg.norm(y[:,3:],axis=1).max()
    finish(fig,out,'07_geometry_long_uncontrolled',
           'Moon-centred orbit projections locate the LSOs, while a continuous unforced trajectory shows how a small initial departure grows over the extended propagation.',
           f'By day {day[-1]:.1f}, peak departures reach {peakp:.3g} km and {peakv:.3g} m/s, with the declared stop condition recorded as “{summary["stop_reason"]}”.')


def validation(orbit,rec,pred,free,out):
    day=np.arange(orbit.n+1)*orbit.dt_s/86400
    fig,axs=plt.subplots(2,1,figsize=(13,9),sharex=True)
    for ax,sl,label in [(axs[0],slice(0,3),'State error [km]'),(axs[1],slice(3,6),'State error [m/s]')]:
        for result,color,name in [(pred,BLUE,'LSO-updated estimate'),(free,RED,'Free model propagation')]:
            norms=np.linalg.norm((result['navigation']-rec.truth)[:,:,sl],axis=-1)
            ax.plot(day,np.sqrt(np.mean(norms**2,axis=0)),color=color,label=name)
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
    axs[1].set_xlabel('Elapsed time [day]')
    err=pred['navigation']-rec.truth
    p=np.sqrt(np.mean(np.sum(err[...,:3]**2,axis=-1)));v=np.sqrt(np.mean(np.sum(err[...,3:]**2,axis=-1)))
    finish(fig,out,'08_heldout_lso_validation',
           'Held-out truth is used only here to compare LSO-updated state estimates with free model propagation after its initial angular update.',
           f'The LSO-updated 3-D RMSE is {p:.3g} km in position and {v:.3g} m/s in velocity, quantifying navigation accuracy without treating estimated states as measurements.')


def residuals(report,out):
    fig,axs=plt.subplots(1,2,figsize=(17,6))
    for c in range(report['acf'].shape[1]):
        axs[0].plot(np.arange(1,report['lags']+1),report['acf'][:,c],label=f'LSO {c//2+1} '+('α' if c%2==0 else 'δ'))
    axs[0].axhline(0,color='grey',lw=.7)
    axs[0].set(xlabel='Actual epoch lag [sample]',ylabel='Standardized angular innovation ACF')
    axs[0].legend(fontsize=9,ncol=2)
    heat=np.nanmax(abs(report['ccf']),axis=0)
    im=axs[1].imshow(heat,aspect='auto',vmin=0,vmax=max(.2,float(np.nanmax(heat))),cmap='viridis')
    axs[1].set(xticks=range(3),xticklabels=['Δv x','Δv y','Δv z'],
               yticks=range(len(heat)),yticklabels=[f'LSO {c//2+1} '+('α' if c%2==0 else 'δ') for c in range(len(heat))])
    fig.colorbar(im,ax=axs[1],label='Maximum |innovation–input CCF| across tested lags')
    rejects=int(np.nansum(report['ljung_box_holm_p']<.05))
    finish(fig,out,'09_optical_innovation_checks',
           'The ACF and residual–input correlation screen the raw angular predictor while preserving visibility gaps and true time lags.',
           f'Holm-adjusted Ljung–Box tests reject {rejects} of {report["eligible_whiteness_tests"]} eligible contiguous runs and the mask-conditioned global CCF p-value is {report["ccf_global_p"]:.3g}, which assesses predictor/noise adequacy rather than controller success.')


def controlled(orbit,trial,no_day,no_y,out):
    fig,axs=plt.subplots(3,1,figsize=(13,12),sharex=True)
    end=trial['day'][-1];use=no_day<=end
    for ax,sl,limit,label in [(axs[0],slice(0,3),10.,'Position departure [km]'),
                              (axs[1],slice(3,6),.1,'Velocity departure [m/s]')]:
        ax.semilogy(no_day[use],np.linalg.norm(no_y[use,sl],axis=1),color=RED,label='Uncontrolled')
        ax.semilogy(trial['day'],np.linalg.norm(trial['output'][:,sl],axis=1),color=BLUE,label='LSO-guided station keeping')
        ax.axhline(limit,color='black',ls='--',label='Required limit')
        ax.set_ylabel(label);ax.legend(fontsize=11);ax.grid(alpha=.2)
    if len(trial['commands']):
        mag=np.linalg.norm(trial['commands'],axis=1)
        points=axs[2].scatter(trial['burn_day'],mag,c=trial['phase_deg'],cmap='twilight',vmin=0,vmax=360,s=35)
        fig.colorbar(points,ax=axs[2],label='Selected burn phase [deg]',pad=.02)
    axs[2].set(xlabel='Elapsed time [day]',ylabel='Commanded impulse [m/s]')
    q=trial['summary']
    finish(fig,out,'10_lso_stationkeeping_corridor',
           'True departure norms evaluate the LSO-guided controller, and coloured impulse markers show the magnitudes and orbital phases it selected.',
           f'Peak position and velocity departures are {q["maximum_position_km"]:.3g} km and {q["maximum_velocity_m_s"]:.3g} m/s, so '+('both limits hold at all evaluated samples.' if q['all_evaluated_samples_inside'] and q['completed'] else 'this run does not establish satisfaction of both corridor limits over the requested span.'))
