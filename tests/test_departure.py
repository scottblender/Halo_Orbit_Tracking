"""Check repeated-reference phase handling and physical departures."""
import numpy as np
from halo_tracking.study import load_config
from halo_tracking.departure import reference_departure
from halo_tracking.dynamics import propagate
from halo_tracking.orbit_reference import correct_halo_reference
from halo_tracking.frames import state_transform


def test_reference_wrap_and_departure():
    """Truth remains rounded; only the repeating nominal is corrected."""
    cfg = load_config()
    result = reference_departure(cfg, periods=2, samples_per_period=20)
    nominal_cfg = correct_halo_reference(cfg)
    nominal_initial = np.asarray(nominal_cfg['orbit']['state'])
    rounded_initial = np.asarray(cfg['orbit']['state'])

    # Both nominal phases reset to the corrected periodic initial condition.
    np.testing.assert_allclose(result['repeated_reference'][0], nominal_initial)
    np.testing.assert_allclose(result['repeated_reference'][20], nominal_initial)
    np.testing.assert_allclose(result['repeated_reference'][40], nominal_initial)
    np.testing.assert_allclose(result['reference_period_states'][0], nominal_initial)
    assert result['diagnostics']['position_closure_km'] < 1e-3
    assert result['diagnostics']['catalog_closure_km'] > 50.

    # A corrected nominal does NOT imply the rounded, uncontrolled truth
    # starts on that orbit. First-period RMS is meaningful, not zero.
    np.testing.assert_allclose(result['states'][0], rounded_initial)
    initial_offset = (np.linalg.norm((rounded_initial - nominal_initial)[:3])
                      * cfg['length_unit_km'])
    np.testing.assert_allclose(result['diagnostics']['initial_nominal_offset_km'],
                               initial_offset)
    np.testing.assert_allclose(result['position_departure_km'][0], initial_offset)
    np.testing.assert_allclose(result['period_position_rms_km'][0],
                               np.sqrt(np.mean(result['position_departure_km'][:20]**2)))
    assert result['period_position_rms_km'][0] > 1.

    # An independent integration confirms there are no state resets.
    expected_end = 2*nominal_cfg['orbit']['period_tu']
    np.testing.assert_allclose(result['times'][-1], expected_end)
    independent = propagate(rounded_initial, [0, expected_end], cfg['mu'])[-1]
    np.testing.assert_allclose(independent, result['states'][-1], atol=1e-9)
    assert len(result['period_position_rms_km']) == 2


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
    assert '0 → 1T' in figure.axes[0].get_title()
    assert '0 → 2T' in figure.axes[1].get_title()
    # Each grid subplot depicts cumulative 0-to-kT motion:
    # 21 samples at T, and 41 samples at 2T.
    assert len(figure.axes[0].lines[1].get_data_3d()[0]) == 21
    assert len(figure.axes[1].lines[1].get_data_3d()[0]) == 41
    np.testing.assert_allclose(
        np.array([axis[0] for axis in figure.axes[0].lines[1].get_data_3d()]),
        np.array([axis[0] for axis in figure.axes[1].lines[1].get_data_3d()]))
    plt.close(figure)
    fixed=period_trajectory_grid(result,cfg,scale_mode='reference')
    for attr in ['get_xlim3d','get_ylim3d','get_zlim3d']:
        np.testing.assert_allclose(getattr(fixed.axes[0],attr)(),
                                   getattr(fixed.axes[1],attr)())
    plt.close(fixed)


def test_rounded_catalog_state_shows_uncontrolled_long_horizon_departure():
    """Guard against silently replacing the unstable seed with a periodic orbit."""
    cfg = load_config()
    assert not cfg['orbit'].get('reference_corrected', False)
    result = reference_departure(cfg, periods=12, samples_per_period=20)
    # Nominal reference closes; the rounded catalog seed does not.
    assert result['diagnostics']['position_closure_km'] < 1e-3
    assert result['diagnostics']['catalog_closure_km'] > 50.
    # Preserve the long-horizon departure regression independently of a
    # particular finite-sample maximum or end-phase alignment.
    assert np.max(result['position_departure_km']) > 500_000.
    nominal_period = correct_halo_reference(cfg)['orbit']['period_tu']
    np.testing.assert_allclose(result['times'][-1], 12*nominal_period)
