"""Check repeated-reference phase handling and physical departures."""
import numpy as np
from halo_tracking.study import load_config
from halo_tracking.departure import reference_departure
from halo_tracking.dynamics import propagate
from halo_tracking.frames import state_transform


def test_reference_wrap_and_departure():
    c=load_config();r=reference_departure(c,periods=2,samples_per_period=20)
    np.testing.assert_allclose(r['repeated_reference'][0],r['repeated_reference'][20])
    np.testing.assert_allclose(r['repeated_reference'][0],r['repeated_reference'][40])
    assert r['period_position_rms_km'][0]<.001
    np.testing.assert_allclose(r['position_departure_km'][20],r['diagnostics']['position_closure_km'],rtol=1e-5)
    independent=propagate(np.array(c['orbit']['state']),[0,r['times'][-1]],c['mu'])[-1]
    np.testing.assert_allclose(independent,r['states'][-1],atol=1e-9)
    assert len(r['period_position_rms_km'])==2


def test_period_trajectory_figures():
    """Create one observer trajectory figure for each period."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from halo_tracking.plotting import period_trajectory_figures
    cfg=load_config()
    result=reference_departure(cfg,periods=2,samples_per_period=20)
    figures=period_trajectory_figures(result,cfg)
    assert list(figures)==['period_01_trajectory','period_02_trajectory']
    for figure in figures.values():
        assert len(figure.axes)==3
        assert len(figure.axes[0].lines)==1
        plt.close(figure)
