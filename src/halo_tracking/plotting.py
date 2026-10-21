"""Shared large-font figures for notebook inspection and export."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .frames import to_mci
from .surface_sensor import site_mci
from .study import physical_history

def style():
    """Apply consistent bold serif text without requiring a proprietary font."""
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':12,'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
        'axes.labelsize':14,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})

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
    """Plot the supplied rotating-frame trajectory and surface site."""
    from .surface_sensor import site_rotating
    style(); cfg=result['config']; L=cfg['length_unit_km']
    r=(result['truth'][:,:3]-[1-cfg['mu'],0,0])*L
    fig=plt.figure(figsize=(9,7),layout='constrained'); ax=fig.add_subplot(projection='3d')
    ax.plot(*r.T,label=cfg['orbit']['family']+' supplied trajectory')
    ax.scatter(*r[result['visible']].T,s=5,label='LSO visible')
    ax.scatter(*(site_rotating(cfg)*L),s=60,label='LSO')
    ax.scatter(0,0,0,s=80,color='gray',label='Moon center')
    gateway=cfg.get('gateway',{})
    if gateway.get('enabled'):
        from .dynamics import propagate
        if gateway.get('state') is None:
            raise ValueError('Supply an independently sourced Gateway state before enabling Gateway')
        g=propagate(np.asarray(gateway['state']),result['times'],cfg['mu'])
        ax.plot(*((g[:,:3]-[1-cfg['mu'],0,0])*L).T,label='Gateway reference')
    ax.set(xlabel='Moon-relative x (km)',ylabel='y (km)',zlabel='z (km)',title='Earth–Moon rotating frame')
    ax.set_box_aspect(np.maximum(np.ptp(r,axis=0),1)); ax.legend(fontsize=10)
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
    """Plot nine physical error components with conditional ±3 sigma bounds."""
    style(); h=physical_history(result,name); cfg=result['config']
    days=result['times']*cfg['time_unit_s']/86400
    sigma=np.c_[np.sqrt(np.maximum(np.diagonal(h['covariance'],axis1=1,axis2=2),0)),h['acceleration_sigma']]
    fig,axes=plt.subplots(3,3,figsize=(14,9),sharex=True,layout='constrained')
    for j,ax in enumerate(axes.flat):
        ax.fill_between(days,-3*sigma[:,j],3*sigma[:,j],color='C0',alpha=.2,label='±3σ')
        if not result[name]['covariance_only']:
            ax.plot(days,h['errors'][:,j],color='black',lw=.8,label='Error')
        ax.set_title(['Position','Velocity','Acceleration'][j//3]+' '+['x','y','z'][j%3])
        ax.set_ylabel(['km','km/s','km/s²'][j//3])
        if j>=6: ax.set_xlabel('Time (days)')
    axes[0,0].legend(); fig.suptitle(name.replace('_',' + ')+' — MCI error and uncertainty')
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
