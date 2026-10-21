"""Shared large-font figures for notebook inspection and export."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .frames import to_mci, state_transform
from .surface_sensor import site_mci
from .layout import style_3d_axes
from .study import physical_history

def style():
    """Apply consistent serif styling at notebook and exported figure sizes."""
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':11,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
        'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,
        'legend.fontsize':10,'legend.frameon':False,'axes.spines.top':False,
        'axes.spines.right':False,'figure.dpi':110,'lines.linewidth':1.6})

def kinematics(result):
    """Plot physical Moon-centered inertial position, velocity and acceleration."""
    style(); cfg=result['config']; days=result['times']*cfg['time_unit_s']/86400
    values=np.array([np.concatenate(to_mci(t,x,cfg)) for t,x in zip(result['times'],result['truth'])])
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    for k,(ax,label) in enumerate(zip(axes,['Position (km)','Velocity (km/s)','Acceleration (km/s²)'])):
        ax.plot(days,values[:,3*k:3*k+3]); ax.set_ylabel(label)
    axes[0].legend(['x','y','z'],ncol=3); axes[-1].set_xlabel('Time (days)')
    fig.suptitle('Nominal trajectory — model Moon-centered inertial frame')
    return fig

def geometry(result):
    """Show one-period mission geometry with equal physical axis scales."""
    from .surface_sensor import site_rotating
    from .dynamics import propagate
    style(); cfg=result['config']; L=cfg['length_unit_km']
    observer=(result['truth'][:,:3]-[1-cfg['mu'],0,0])*L
    target=None
    if cfg.get('gateway',{}).get('enabled') and cfg['gateway'].get('state') is not None:
        states=result.get('gateway_truth')
        if states is None:states=propagate(np.asarray(cfg['gateway']['state']),result['times'],cfg['mu'])
        target=(states[:,:3]-[1-cfg['mu'],0,0])*L
    points=np.vstack([observer, np.zeros(3), site_rotating(cfg)*L]+([] if target is None else [target]))
    span=np.ptp(points,axis=0).max()
    exponent=3*int(np.floor(np.log10(max(span,1.))/3))
    scale=10.**exponent
    unit='km' if exponent==0 else f'10$^{{{exponent}}}$ km'
    fig=plt.figure(figsize=(8.2,6.8),layout='constrained')
    ax=fig.add_subplot(projection='3d');ax.set_proj_type('ortho')
    ax.plot(*(observer/scale).T,color='#245b80',label='Halo observer')
    if target is not None:ax.plot(*(target/scale).T,color='#ca7139',label='Lunar Gateway')
    # The physical Moon radius is retained; it is not enlarged to imply range.
    u=np.linspace(0,2*np.pi,32);v=np.linspace(0,np.pi,16)
    radius=cfg['site']['radius_km']/scale
    ax.plot_surface(radius*np.outer(np.cos(u),np.sin(v)),radius*np.outer(np.sin(u),np.sin(v)),
                    radius*np.outer(np.ones_like(u),np.cos(v)),color='#9aa0a6',alpha=.8,linewidth=0)
    ax.scatter([],[],[],color='#9aa0a6',s=24,label='Moon')
    ax.scatter(*(site_rotating(cfg)*L/scale),marker='*',s=55,color='#4b7354',label='Lunar surface sensor',depthshade=False)
    ax.scatter(*(observer[0]/scale),color='#245b80',s=20,depthshade=False)
    if target is not None:ax.scatter(*(target[0]/scale),color='#ca7139',s=20,depthshade=False)
    center=(points.min(axis=0)+points.max(axis=0))/(2*scale)
    half=max(span/scale/2,np.finfo(float).eps)*1.06
    for setter,value in zip([ax.set_xlim3d,ax.set_ylim3d,ax.set_zlim3d],center):setter(value-half,value+half)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-52)
    ax.set(xlabel=f'x ({unit})',ylabel=f'y ({unit})',zlabel=f'z ({unit})')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((1.,1.,1.,0.))
        axis.pane.set_edgecolor((1.,1.,1.,0.))
        axis._axinfo['grid'].update(color=(.75,.77,.79,.5),linewidth=.5)
    ax.legend(loc='upper left',ncol=2,bbox_to_anchor=(0.,1.02),columnspacing=1.2)
    fig.suptitle('Moon-centered rotating frame',fontsize=13,fontweight='bold')
    style_3d_axes(ax,label_gap_fraction=.016)
    return fig

def measurement_history(result):
    """Plot angular measurements and elevation; gaps remain blank."""
    from .measurements import predict,ARCSEC
    style(); cfg=result['config']; days=result['times']*cfg['time_unit_s']/86400
    nominal=np.array([predict(t,x,cfg,True) for t,x in zip(result['times'],result['truth'])])
    fig,axes=plt.subplots(5,1,figsize=(10,12),sharex=True,layout='constrained')
    for j,label in enumerate(['Alpha (deg)','Delta (deg)','Alpha rate (arcsec/s)','Delta rate (arcsec/s)']):
        factor=180/np.pi if j<2 else 1/ARCSEC
        axes[j].plot(days,nominal[:,j]*factor,color='0.6',label='Geometric truth')
        if result['measurements'] is not None:
            axes[j].plot(days,result['measurements'][:,j]*factor,'.',ms=2,label='Available measurements')
        axes[j].set_ylabel(label)
    axes[0].legend(); axes[-1].plot(days,np.rad2deg(result['elevation']))
    axes[-1].axhline(cfg['site']['minimum_elevation_deg'],ls='--',color='black')
    axes[-1].set(ylabel='Elevation (deg)',xlabel='Time (days)')
    return fig

def estimation_errors(result,name='angles'):
    """Plot component errors with readable units and separated row labels."""
    style(); h=physical_history(result,name); cfg=result['config']
    days=result['times']*cfg['time_unit_s']/86400
    sigma=np.c_[np.sqrt(np.maximum(np.diagonal(h['covariance'],axis1=1,axis2=2),0)),h['acceleration_sigma']]
    fig,axes=plt.subplots(3,3,figsize=(11,8),sharex=True,layout='constrained')
    fig.get_layout_engine().set(hspace=.12,wspace=.12)
    factors=[1.,1000.,1e9]
    labels=['Position error\n(km)','Velocity error\n(m/s)','Acceleration error\n(μm/s²)']
    for j,ax in enumerate(axes.flat):
        row=j//3;factor=factors[row]
        ax.fill_between(days,-3*sigma[:,j]*factor,3*sigma[:,j]*factor,color='#80a9c4',alpha=.3,label='±3σ (local)')
        if not result[name]['covariance_only']:
            ax.plot(days,h['errors'][:,j]*factor,color='#273c4d',lw=1.,label='Actual error')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        if row==0:ax.set_title(['x','y','z'][j%3])
        if j%3==0:ax.set_ylabel(labels[row])
        if row==2:ax.set_xlabel('Time (days)')
    axes[0,0].legend(fontsize=9)
    title='Angles only' if name=='angles' else 'Angles and angular rates'
    fig.suptitle('Observer state errors — '+title,fontsize=13,fontweight='bold')
    return fig

def uncertainty_comparison(result):
    """Compare root-sum-square position and velocity standard deviations."""
    style(); days=result['times']*result['config']['time_unit_s']/86400
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for name in ['angles','angles_rates']:
        P=physical_history(result,name)['covariance']
        for k,ax in enumerate(axes):
            rss=np.sqrt(np.maximum(np.trace(P[:,k*3:k*3+3,k*3:k*3+3],axis1=1,axis2=2),0))
            ax.semilogy(days,rss,label=name.replace('_',' + '))
    axes[0].set_ylabel('Position RSS σ (km)'); axes[1].set_ylabel('Velocity RSS σ (km/s)')
    axes[1].set_xlabel('Time (days)'); axes[0].legend()
    return fig

def export(fig,path):
    """Save a figure with a white background."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=200,facecolor='white',bbox_inches='tight')


def horizon_comparison(result):
    """Plot continuous forecast uncertainty and exact period checkpoints."""
    style()
    cfg = result['config']
    times = result['times']
    periods = times / cfg['orbit']['period_tu']
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout='constrained')
    from .frames import state_transform, acceleration_jacobian
    for name in ['angles', 'angles_rates']:
        values = []
        for t, x, P in zip(times, result['truth'], result[name]['covariance']):
            B = state_transform(t, cfg); C = B @ P @ B.T
            G = acceleration_jacobian(t, x, cfg); Ca = G @ P @ G.T
            values.append([np.sqrt(max(np.trace(C[:3,:3]), 0)),
                           1000*np.sqrt(max(np.trace(C[3:,3:]), 0)),
                           1000*np.sqrt(max(np.trace(Ca), 0))])
        values = np.asarray(values)
        for j, ax in enumerate(axes):
            line, = ax.semilogy(periods, values[:,j], label=name.replace('_',' + '))
            for row in result['horizon_rows']:
                if row['measurement_case'] == name:
                    key = ['position_rss_sigma_km','velocity_rss_sigma_m_s','acceleration_rss_sigma_m_s2'][j]
                    ax.scatter(row['periods'], row[key], color=line.get_color(), s=22)
    for ax in axes:
        for p in result['horizon_periods']:
            ax.axvline(p, color='gray', alpha=.3, ls=':')
    axes[0].set_ylabel('Position RSS σ (km)')
    axes[1].set_ylabel('Velocity RSS σ (m/s)')
    axes[2].set(ylabel='Acceleration RSS σ (m/s²)', xlabel='Elapsed nominal periods')
    axes[0].legend()
    fig.suptitle('Nominal covariance forecast — continuous LSO tracking, no stationkeeping')
    return fig


def gateway_pointing_uncertainty(history,cfg):
    """Plot observer-induced pointing and angular-rate uncertainty only."""
    style()
    periods=history['times']/cfg['orbit']['period_tu']
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for name in ['angles','angles_rates']:
        axes[0].semilogy(periods,history[name]['pointing_rss_sigma_arcsec'],label=name.replace('_',' + '))
        axes[1].semilogy(periods,history[name]['angular_rate_rss_sigma_arcsec_s'],label=name.replace('_',' + '))
    axes[0].set_ylabel('Observer-induced pointing RSS σ (arcsec)')
    axes[1].set(ylabel='Angular-rate RSS σ (arcsec/s)',xlabel='Elapsed observer nominal periods')
    axes[0].legend();fig.suptitle('Effect of observer orbit covariance on known-Gateway optical geometry')
    return fig


def gateway_comparison_figures(result):
    """Return separate actual-error and conditional-uncertainty comparisons."""
    from .gateway_comparison import comparison_metrics
    style(); metrics=comparison_metrics(result)
    days=result['times']*result['config']['time_unit_s']/86400
    names=['perfect','lso_estimated']; labels=['Perfect observer knowledge','Observer estimated from LSO']
    colors=['#245b80','#ca7139']; figures={}
    for kind,title,keys in [
        ('errors','Gateway estimation errors — one shared noise realization',['position_error_km','velocity_error_m_s']),
        ('uncertainty','Gateway predicted uncertainty — local covariance',['position_forecast_rss_sigma_km','velocity_forecast_rss_sigma_m_s'])]:
        fig,axes=plt.subplots(2,1,figsize=(9.4,6.2),sharex=True,layout='constrained')
        fig.get_layout_engine().set(hspace=.10)
        for name,label,color in zip(names,labels,colors):
            for j,ax in enumerate(axes):ax.semilogy(days,metrics[name][keys[j]],color=color,label=label)
        axes[0].set_ylabel('Position '+('error' if kind=='errors' else 'RSS σ')+' (km)')
        axes[1].set_ylabel('Velocity '+('error' if kind=='errors' else 'RSS σ')+' (m/s)')
        axes[1].set_xlabel('Time (days)')
        fig.suptitle(title,fontsize=12,fontweight='bold');axes[0].legend(loc='upper right')
        figures[kind]=fig
    return figures


def observer_knowledge(result):
    """Explain observer errors and the timing of LSO updates in one figure."""
    style();cfg=result['config'];days=result['times']*cfg['time_unit_s']/86400
    errors=[];sigma=[]
    for k,t in enumerate(result['times']):
        B=state_transform(t,cfg)
        error=B@(result['lso_estimated']['observer_state'][k]-result['truth'][k])
        C=B@result['lso_estimated_forecast']['observer_covariance'][k]@B.T
        errors.append([np.linalg.norm(error[:3]),1000*np.linalg.norm(error[3:])])
        sigma.append([np.sqrt(max(np.trace(C[:3,:3]),0)),1000*np.sqrt(max(np.trace(C[3:,3:]),0))])
    errors=np.array(errors);sigma=np.array(sigma)
    fig,axes=plt.subplots(2,1,figsize=(9.4,6.2),sharex=True,layout='constrained')
    for j,ax in enumerate(axes):
        ax.plot(days,errors[:,j],color='#ca7139',label='Observer error: one realization')
        ax.plot(days,sigma[:,j],color='#245b80',ls='--',label='Predicted RSS σ: local covariance')
        # Shade the intervals bounded by the available scheduled samples.
        visible=result['lso_available'];edges=np.diff(np.r_[False,visible,False].astype(int))
        for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            ax.axvspan(days[start],days[min(end,len(days)-1)],color='#80a78a',alpha=.14)
        ax.set_ylabel(['Position (km)','Velocity (m/s)'][j])
    axes[0].legend(loc='upper left',fontsize=9);axes[1].set_xlabel('Time (days)')
    fig.suptitle('LSO knowledge of the halo observer — shaded interval contains LSO observations',fontsize=12,fontweight='bold')
    return fig
