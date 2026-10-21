"""Paired Monte Carlo validation of the conditional EKF covariance."""
from copy import deepcopy
import numpy as np
from .study import make_truth,physical_history
from .measurements import simulate
from .estimation import run_ekf

def monte_carlo(cfg,trials=100,progress=None):
    """Return epochwise RMSE, component 3-sigma coverage, NEES and NIS."""
    if trials<2:
        raise ValueError('Use at least two trials')
    base=make_truth(cfg); output={}
    for name,rates in [('angles',False),('angles_rates',True)]:
        errors=[]; cover=[]; nees=[]; nis=[]
        for trial in range(trials):
            c=deepcopy(cfg); c['simulation']['seed']=cfg['simulation']['seed']+1000*trial
            z=simulate(base['times'],base['truth'],base['visible'],c)
            fit=run_ekf(base['times'],base['truth'],base['visible'],z,c,rates)
            result={**base,'config':c,name:fit}; h=physical_history(result,name)
            errors.append(h['errors'][:,:6]); sig=np.sqrt(np.maximum(np.diagonal(h['covariance'],axis1=1,axis2=2),0))
            cover.append(np.abs(h['errors'][:,:6])<=3*sig)
            e=fit['state']-base['truth']
            nees.append([v@np.linalg.solve(P,v) for v,P in zip(e,fit['covariance'])]); nis.append(fit['nis'])
            if progress is not None:
                progress(name,trial+1,trials)
        errors=np.array(errors)
        output[name]=dict(position_rmse_km=np.sqrt(np.mean(np.sum(errors[:,:,:3]**2,axis=2),axis=0)),
            velocity_rmse_km_s=np.sqrt(np.mean(np.sum(errors[:,:,3:]**2,axis=2),axis=0)),
            component_3sigma_coverage=np.mean(cover,axis=0),mean_nees=np.mean(nees,axis=0),
            nis=np.array(nis))
    return dict(times=base['times'],visible=base['visible'],trials=trials,**output)
