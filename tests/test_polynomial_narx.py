import numpy as np
from pathlib import Path
import joblib
from halo_tracking.frames import state_transform
from halo_tracking.study import load_config
from halo_tracking.polynomial_narx import state_input,regressor,usable,metrics,PolynomialController
from halo_tracking.polynomial_narx import fit_polynomial_narx, predict
from halo_tracking.polynomial_experiment import run_polynomial_pair


def test_rotating_inputs_exactly_invert_mci_departure():
    cfg = load_config()
    delta = np.array([1e-5,-2e-5,3e-5,4e-6,-5e-6,6e-6])
    for t in (0.,.7,12.):
        nav = state_transform(t,cfg) @ delta
        nav[3:] *= 1000
        x = state_input(nav,t,2.,cfg)
        np.testing.assert_allclose(x[:3],delta[:3]*cfg['length_unit_km'],atol=1e-12)
        np.testing.assert_allclose(x[3:6],delta[3:]*1000*cfg['length_unit_km']/cfg['time_unit_s'],atol=1e-12)


def test_regressor_excludes_current_and_future_outputs():
    x = np.arange(80.).reshape(10,8); y = np.arange(30.).reshape(10,3)
    before = regressor(x,y,5,4,.125)
    y[5:] = -999; x[6:] = -999
    np.testing.assert_array_equal(before,regressor(x,y,5,4,.125))
    assert len(before)==45
    assert before[-1]==.125


def test_unknown_feedback_invalidates_history_and_nonfinite_is_counted():
    x = np.ones((1,10,8)); known = np.ones((1,10),bool); known[0,4]=False
    mask = usable(x,known,4)
    assert not mask[0,:9].any()
    assert mask[0,9]
    p = np.zeros((1,2,3)); p[0,1] = np.nan
    m = metrics(p,np.zeros_like(p),np.ones((1,2),bool))
    assert m['nonfinite_burns']==1 and m['scored_burns']==1


def test_no_burn_does_not_consult_future_reference_vectors():
    cfg = load_config()
    schedule = dict(maneuver_requested=np.zeros(6,bool),predicted_valid=np.ones(6,bool),
        maneuver_rejected=np.zeros(6,bool))
    c = PolynomialController(dict(history_steps=4),cfg,2.,schedule)
    for k in range(6):
        command = c.command(np.zeros(6),float(k),k,10.)
        assert not command['required']
        np.testing.assert_array_equal(c.y[-1],np.zeros(3))


def test_fitted_sparse_polynomial_roundtrip_and_matched_schedule(tmp_path):
    cfg = load_config(); cfg['narx']['closed_loop_measurement_cadence_s']=21600.
    dataset = Path(__file__).resolve().parents[1]/'artifacts/neural_narx/teacher_training'
    b,report = fit_polynomial_narx(dataset,cfg,tmp_path/'fit')
    assert report['splits']['training']['usable_burns']==41
    assert report['splits']['test']['recursive']['scored_burns']==9
    assert all(np.count_nonzero(m.coef_)<=report['selected_max_terms_per_component'] for m in b['models'])
    restored = joblib.load(tmp_path/'fit/polynomial_narx.joblib')
    row = b['input_scaler'].mean_[None]
    np.testing.assert_allclose(predict(restored,row),predict(b,row))
    e,summary = run_polynomial_pair(cfg,b,tmp_path/'orbit',days=6.)
    assert e['STM']['completed_full_span'] and e['Polynomial NARX']['completed_full_span']
    assert e['STM']['observer_truth_position_departure_km'][0]==e['Polynomial NARX']['observer_truth_position_departure_km'][0]
    np.testing.assert_array_equal(e['STM']['maneuver_requested'],e['Polynomial NARX']['maneuver_requested'])
    L = b['history_steps']
    np.testing.assert_allclose(e['STM']['applied_delta_v_rotating_m_s'][:L],e['Polynomial NARX']['applied_delta_v_rotating_m_s'][:L])
