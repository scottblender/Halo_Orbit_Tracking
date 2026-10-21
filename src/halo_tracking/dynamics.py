"""Normalized Earth-Moon barycentric rotating CR3BP dynamics."""
import numpy as np
from scipy.integrate import solve_ivp

def rhs(t, state, mu):
    """Return [velocity, rotating acceleration] in LU and TU."""
    r, v = np.asarray(state)[:3], np.asarray(state)[3:6]
    d1 = r + [mu, 0, 0]
    d2 = r - [1-mu, 0, 0]
    a = -(1-mu)*d1/np.linalg.norm(d1)**3-mu*d2/np.linalg.norm(d2)**3
    a += [r[0]+2*v[1], r[1]-2*v[0], 0]
    return np.r_[v, a]

def jacobian(state, mu):
    """Return the analytic six-state dynamics Jacobian."""
    r = np.asarray(state)[:3]
    U = np.diag([1., 1., 0.])
    for mass, d in [(1-mu, r+[mu,0,0]), (mu, r-[1-mu,0,0])]:
        n = np.linalg.norm(d)
        U += mass*(3*np.outer(d,d)/n**5-np.eye(3)/n**3)
    A = np.zeros((6,6))
    A[:3,3:] = np.eye(3)
    A[3:,:3] = U
    A[3:,3:] = [[0,2,0],[-2,0,0],[0,0,0]]
    return A

def propagate(state, times, mu):
    """Propagate a state at times[0] to an increasing sequence of epochs."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times)<1 or np.any(np.diff(times)<=0):
        raise ValueError('times must be a nonempty strictly increasing vector')
    if len(times)==1:
        return np.asarray(state, dtype=float)[None,:]
    sol = solve_ivp(lambda t,x:rhs(t,x,mu), (times[0],times[-1]),state,
                    t_eval=times,method='DOP853',rtol=2e-11,atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T

def propagate_step(state, start, end, mu, acceleration_psd=0.):
    """Integrate state, STM, and continuous white-acceleration covariance."""
    if end==start:
        return state.copy(), np.eye(6), np.zeros((6,6))
    def augmented(t,y):
        A = jacobian(y[:6],mu)
        Phi, W = y[6:42].reshape(6,6), y[42:].reshape(6,6)
        L = np.diag([0.,0.,0.,acceleration_psd,acceleration_psd,acceleration_psd])
        return np.r_[rhs(t,y[:6],mu),(A@Phi).ravel(),(A@W+W@A.T+L).ravel()]
    y0 = np.r_[state,np.eye(6).ravel(),np.zeros(36)]
    sol = solve_ivp(augmented,(start,end),y0,method='DOP853',rtol=2e-10,atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    y=sol.y[:,-1]
    return y[:6],y[6:42].reshape(6,6),y[42:].reshape(6,6)

def jacobi(state, mu):
    """Evaluate the Jacobi integral for one state."""
    r,v = state[:3],state[3:]
    return r[0]**2+r[1]**2+2*(1-mu)/np.linalg.norm(r+[mu,0,0])+2*mu/np.linalg.norm(r-[1-mu,0,0])-v@v
