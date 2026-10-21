"""Independent numerical checks of dynamics, frames, rates, and estimation."""
from copy import deepcopy
import numpy as np
from numpy.testing import assert_allclose
from halo_tracking.study import load_config,run_study
from halo_tracking.dynamics import rhs,jacobian,propagate,jacobi,propagate_step
from halo_tracking.frames import to_mci,acceleration_jacobian
from halo_tracking.measurements import predict,measurement_jacobian,wrap
from halo_tracking.surface_sensor import site_rotating,elevation

def test_dynamics_stm_and_jacobi():
    c=load_config(); x=np.array(c['orbit']['state']); eps=1e-6
    numeric=np.column_stack([(rhs(0,x+eps*e,c['mu'])-rhs(0,x-eps*e,c['mu']))/(2*eps) for e in np.eye(6)])
    assert_allclose(jacobian(x,c['mu']),numeric,rtol=1e-7,atol=1e-8)
    ts=np.linspace(0,.2,30); hist=propagate(x,ts,c['mu'])
    assert np.ptp([jacobi(y,c['mu']) for y in hist])<1e-9
    _,Phi,Q=propagate_step(x,0,.05,c['mu'],1e-12)
    d=np.array([1,-1,1,-1,1,-1])*1e-7
    assert_allclose(propagate(x+d,[0,.05],c['mu'])[-1]-propagate(x,[0,.05],c['mu'])[-1],Phi@d,atol=2e-12)
    assert np.linalg.eigvalsh(Q).min()>0

def test_inertial_derivatives_and_angular_rates():
    c=load_config(); x=np.array(c['orbit']['state']); t=.13; dt=1e-5
    xm=propagate(x,[t,t+dt],c['mu'])[-1]
    xp=propagate(x,[t,t+2*dt],c['mu'])[-1]
    r0,v0,_=to_mci(t,x,c); rm,vm,am=to_mci(t+dt,xm,c); rp,vp,_=to_mci(t+2*dt,xp,c)
    assert_allclose((rp-r0)/(2*dt*c['time_unit_s']),vm,rtol=1e-7,atol=1e-8)
    assert_allclose((vp-v0)/(2*dt*c['time_unit_s']),am,rtol=1e-7,atol=1e-11)
    difference=predict(t+2*dt,xp,c)-predict(t,x,c); difference[0]=wrap(difference[0])
    assert_allclose(difference/(2*dt*c['time_unit_s']),predict(t+dt,xm,c,True)[2:],rtol=1e-7,atol=1e-12)
    G=acceleration_jacobian(t,x,c)
    assert_allclose(G[:,3:],0,atol=1e-20)

def test_measurement_jacobian_and_site():
    c=load_config(); x=np.array(c['orbit']['state']); d=np.arange(1,7)*1e-8
    H=measurement_jacobian(.1,x,c,True)
    diff=predict(.1,x+d,c,True)-predict(.1,x,c,True);diff[0]=wrap(diff[0])
    assert_allclose(diff,H@d,atol=1e-12,rtol=1e-4)
    site=site_rotating(c); moon=np.array([1-c['mu'],0,0])
    overhead=np.r_[moon+site*2,np.zeros(3)]
    assert_allclose(elevation(overhead,c),np.pi/2,atol=2e-7)

def test_ekf_gaps_and_rate_information():
    c=deepcopy(load_config());c['simulation']['periods']=.05;c['simulation']['cadence_s']=1000
    c['site']['minimum_elevation_deg']=90
    r=run_study(c,covariance_only=True)
    assert not r['visible'].any()
    assert_allclose(r['angles']['covariance'],r['angles_rates']['covariance'])
    c['site']['minimum_elevation_deg']=-90 # Synthetic all-visible algebra check only.
    r=run_study(c,covariance_only=True)
    P=r['angles']['covariance'][-1]-r['angles_rates']['covariance'][-1]
    assert np.linalg.eigvalsh(P).min()>-1e-17
    assert np.linalg.eigvalsh(r['angles_rates']['covariance'][-1]).min()>0
