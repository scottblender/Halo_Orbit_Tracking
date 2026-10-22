"""One-period CR3BP reference for Gateway."""
import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import rhs


class GatewayReference:
    """Evaluate Gateway's fixed nominal orbit at any epoch."""
    def __init__(self, cfg):
        self.period = float(cfg['gateway']['period_tu'])
        if self.period <= 0:
            raise ValueError('Gateway period must be positive')
        self.initial = np.asarray(cfg['gateway']['state'], dtype=float)
        if self.initial.shape != (6,) or not np.isfinite(self.initial).all():
            raise ValueError('Invalid Gateway initial state')
        sol = solve_ivp(lambda t, x: rhs(t, x, cfg['mu']), (0, self.period),
                        self.initial, method='DOP853', dense_output=True,
                        rtol=2e-11, atol=2e-13)
        if not sol.success:
            raise RuntimeError(sol.message)
        self.solution = sol.sol
        delta = self.solution(self.period) - self.initial
        self.position_closure_km = float(np.linalg.norm(delta[:3]) * cfg['length_unit_km'])
        self.velocity_closure_m_s = float(
            1000 * np.linalg.norm(delta[3:]) * cfg['length_unit_km'] / cfg['time_unit_s'])
        if self.position_closure_km > 0.001 or self.velocity_closure_m_s > 0.001:
            raise ValueError('Gateway reference is not sufficiently periodic')

    def states(self, times):
        """Return rotating-frame states, repeating at the Gateway period."""
        times = np.asarray(times, dtype=float)
        if np.any(~np.isfinite(times)) or np.any(times < 0):
            raise ValueError('Epochs must be finite and nonnegative')
        phase = np.mod(times, self.period)
        return np.asarray(self.solution(phase)).T

    def closure(self):
        """Return one-period closure diagnostics."""
        return dict(position_km=self.position_closure_km,
                    velocity_m_s=self.velocity_closure_m_s,
                    gateway_period_tu=self.period)
