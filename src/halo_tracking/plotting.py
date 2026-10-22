"""Shared large-font figures for notebook inspection and export."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .frames import to_mci, state_transform
from .surface_sensor import site_mci
from .network import sensor_configs
from .layout import style_3d_axes

def style():
    """Apply consistent serif styling at notebook and exported figure sizes."""
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':11,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
        'axes.labelsize':11,'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,
        'legend.fontsize':10,'legend.frameon':False,'axes.spines.top':False,
        'axes.spines.right':False,'figure.dpi':110,'lines.linewidth':1.6})

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
    for local in sensor_configs(cfg):
        ax.scatter(*(site_rotating(local)*L/scale),marker='*',s=55,
                   label=local['site'].get('name','LSO'),depthshade=False)
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

def export(fig,path):
    """Save a figure with a white background."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=200,facecolor='white',bbox_inches='tight')


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


def state_histories(result, body):
    """Compare Moon-centered inertial position, velocity and acceleration."""
    style()
    cfg = result['config']
    times = result['times']
    days = times * cfg['time_unit_s'] / 86400
    if body == 'observer':
        histories = [result['truth'], result['lso_estimated']['observer_state']]
        labels = ['Perfect observer (truth)', 'LSO-predicted observer']
        title = 'Halo observer'
        colors = ['#245b80', '#ca7139']
        styles = ['-', '--']
    elif body == 'gateway':
        histories = [result['gateway_truth'], result['perfect']['state'], result['lso_estimated']['state']]
        labels = ['Gateway truth', 'Perfect-observer tracking', 'LSO-observer tracking']
        title = 'Lunar Gateway'
        colors = ['#41464b', '#245b80', '#ca7139']
        styles = ['-', '--', ':']
    else:
        raise ValueError('body must be observer or gateway')
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.8), sharex=True, layout='constrained')
    fig.get_layout_engine().set(hspace=.08, wspace=.08)
    factors = [1e-3, 1e3, 1e6]
    units = ['Position (10³ km)', 'Velocity (m/s)', 'Acceleration (mm/s²)']
    for history, label, color, linestyle in zip(histories, labels, colors, styles):
        values = np.array([np.concatenate(to_mci(t, x, cfg)) for t, x in zip(times, history)])
        for row in range(3):
            for col in range(3):
                axes[row, col].plot(days, values[:, 3*row+col]*factors[row],
                                    color=color, ls=linestyle, label=label)
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.grid(alpha=.18)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.locator_params(axis='y', nbins=5)
            if row == 0: ax.set_title(['x', 'y', 'z'][col])
            if col == 0: ax.set_ylabel(units[row])
            if row == 2: ax.set_xlabel('Time (days)')
    fig.suptitle(title + ' — Moon-centered inertial state histories', fontsize=13)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center',
               ncol=len(labels), fontsize=10)
    return fig


def departure_figures(result, cfg):
    """Show instantaneous departure and equal-phase period RMS growth."""
    style()
    periods = result['times']/cfg['orbit']['period_tu']
    figures = {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    for ax, key, label in zip(axes, ['position_departure_km','velocity_departure_m_s'],
                             ['Position departure (km)', 'Velocity departure (m/s)']):
        values = result[key].copy()
        # The first traversal is the reference, so suppress numerical near-zero differences.
        values[periods < 1] = np.nan
        ax.semilogy(periods, np.maximum(values, 1e-12), color='#245b80')
        for k in range(1, result['diagnostics']['periods']+1):
            ax.axvline(k, color='0.7', lw=.5, alpha=.4)
        ax.set_ylabel(label); ax.grid(axis='y', alpha=.2)
    axes[-1].set_xlabel('Elapsed nominal periods')
    fig.suptitle('Uncontrolled departure from the repeated one-period reference')
    figures['departure'] = fig
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    periods = np.arange(1, len(result['period_position_rms_km'])+1)
    for ax, key, label in zip(axes, ['period_position_rms_km','period_velocity_rms_m_s'],
                             ['Period RMS position departure (km)', 'Period RMS velocity departure (m/s)']):
        ax.semilogy(periods[1:], result[key][1:], 'o-', color='#ca7139', ms=4)
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Period number (equal orbital-phase samples)')
    axes[-1].set_xticks(np.arange(2, len(periods)+1, 2))
    fig.suptitle('Growth across periods — first reference traversal excluded')
    figures['period_growth'] = fig
    return figures


def period_trajectory_figures(result, cfg):
    """Plot the observer trajectory separately for each period."""
    style()
    count = result['diagnostics']['periods']
    n = result['diagnostics']['samples_per_period']
    moon = np.array([1 - cfg['mu'], 0., 0.])
    length = cfg['length_unit_km']
    views = [('x', 'y', 0, 1), ('x', 'z', 0, 2), ('y', 'z', 1, 2)]
    figures = {}
    for k in range(count):
        observer = (result['states'][k*n:(k+1)*n+1, :3] - moon) * length
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), layout='constrained')
        for ax, (a, b, i, j) in zip(axes, views):
            ax.plot(observer[:, i], observer[:, j], color='#245b80')
            ax.scatter(observer[0, i], observer[0, j], color='#245b80', s=24)
            ax.scatter(observer[-1, i], observer[-1, j], marker='x', color='#ca7139', s=35)
            ax.set(xlabel=f'{a} (km)', ylabel=f'{b} (km)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.2)
        fig.suptitle(f'Observer orbit — period {k + 1}', fontsize=12, fontweight='bold')
        figures[f'period_{k+1:02d}_trajectory'] = fig
    return figures
