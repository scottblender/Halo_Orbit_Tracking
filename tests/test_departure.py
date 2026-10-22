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


def test_period_trajectory_grid():
    """Local scale reveals divergence; fixed scale uses consistent axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from halo_tracking.plotting import period_trajectory_grid
    cfg=load_config()
    result=reference_departure(cfg,periods=2,samples_per_period=20)
    figure=period_trajectory_grid(result,cfg)
    assert len(figure.axes)==2
    assert all(axis.name=='3d' for axis in figure.axes)
    assert all(len(axis.lines)==2 for axis in figure.axes)
    assert not figure.axes[0].get_xticklabels()
    assert not figure.axes[1].get_xticklabels()
    assert 'Period 01' in figure.axes[0].get_title()
    assert 'Period 02' in figure.axes[1].get_title()
    plt.close(figure)
    fixed=period_trajectory_grid(result,cfg,scale_mode='reference')
    for attr in ['get_xlim3d','get_ylim3d','get_zlim3d']:
        np.testing.assert_allclose(getattr(fixed.axes[0],attr)(),
                                   getattr(fixed.axes[1],attr)())
    plt.close(fixed)
