"""Sparse quadratic NARX for delta-v conditional on a supplied burn schedule.

No timing classifier, neural network, or reference delta-v fallback after warmup.
"""
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuit
import joblib
from .two_stage_narx import _load
from .frames import rotation, W


def state_input(nav, t, period, cfg):
    """MCI km/m/s departure -> rotating km/m/s departure plus orbit phase.

    Remove the rotating-frame transport velocity, not just rotate the vector.
    """
    r = rotation(t).T @ np.asarray(nav[:3])
    v = rotation(t).T @ np.asarray(nav[3:]) - 1000/cfg['time_unit_s'] * W @ r
    phase = 2*np.pi*t/period
    return np.r_[r, v, np.sin(phase), np.cos(phase)]


def labels(data):
    required = data['observer_maneuver_required'].astype(bool)
    dv = data['observer_target_delta_v_rotating_m_s']
    known = ~required | (data['observer_maneuver_valid'] & np.isfinite(dv).all(-1))
    active = required & known
    return np.where(active[...,None], dv, 0.), known, active


def inputs(data, cfg):
    times = data['decision_times_tu']
    period = times[-1]/cfg['narx']['arc_duration_periods']
    x = np.array([[state_input(nav,t,period,cfg) for nav,t in zip(arc,times)]
                  for arc in data['observer_navigation_error_km_m_s']])
    return x, period


def regressor(x, y, k, L, delay_fraction):
    return np.r_[x[k-L+1:k+1].ravel(), y[k-L:k].ravel(), delay_fraction]


def usable(x, known, L):
    mask = np.zeros_like(known)
    for k in range(L,x.shape[1]):
        mask[:,k] = known[:,k-L:k+1].all(-1) & np.isfinite(x[:,k-L+1:k+1]).all((1,2))
    return mask


def predict(bundle, rows):
    rows = np.atleast_2d(rows)
    with np.errstate(over='raise', invalid='raise'):
        p = bundle['poly'].transform(bundle['input_scaler'].transform(rows))
        p = bundle['term_scaler'].transform(p)[:,bundle['variable_terms']]
        return np.column_stack([model.predict(p) for model in bundle['models']])


def rollout(bundle, x, y, active, delays, period, reference_history=False):
    """Same supplied schedule for both modes; zero past commands at start.

    The first L commands are common STM warmup. Predictions never feed back
    test target vectors after warmup in the recursive evaluation.
    """
    L = bundle['history_steps']
    result = np.zeros_like(y)
    failures = []
    for a in range(len(x)):
        result[a,:L] = y[a,:L]
        issued = np.zeros_like(y[a]); issued[:L] = y[a,:L]
        for k in range(L,x.shape[1]):
            if not active[a,k]:
                continue
            feedback = y[a] if reference_history else issued
            try:
                result[a,k] = predict(bundle,regressor(x[a],feedback,k,L,delays[a,k]/period))[0]
                if not np.isfinite(result[a,k]).all():
                    raise FloatingPointError('nonfinite prediction')
                if np.linalg.norm(result[a,k])<=bundle['maximum_impulse_m_s']:
                    issued[k] = result[a,k]
            except (ValueError, FloatingPointError) as exc:
                result[a,k:] = np.nan
                failures.append(dict(arc=a,decision=k,reason=str(exc)))
                break
    return result, failures


def metrics(p, y, mask):
    finite = np.isfinite(p).all(-1)
    selected = mask & finite
    if not selected.any():
        return dict(scored_burns=0,nonfinite_burns=int(mask.sum()),vector_rmse_m_s=None)
    e = p[selected]-y[selected]
    return dict(scored_burns=int(selected.sum()),nonfinite_burns=int((mask & ~finite).sum()),
        vector_rmse_m_s=float(np.sqrt(np.mean(np.sum(e*e,-1)))),
        component_rmse_m_s=np.sqrt(np.mean(e*e,0)).tolist(),
        component_bias_m_s=e.mean(0).tolist(),
        maximum_vector_error_m_s=float(np.max(np.linalg.norm(e,axis=-1))))


def fit_polynomial_narx(dataset_dir, cfg, output_dir):
    data = {s:_load(Path(dataset_dir)/(s+'.npz')) for s in ('training','validation','test')}
    x = {}; y = {}; active = {}; masks = {}; periods = {}; delays = {}
    L = cfg['narx']['history_steps']
    for s,d in data.items():
        x[s],periods[s] = inputs(d,cfg)
        y[s],known,active[s] = labels(d)
        masks[s] = usable(x[s],known,L) & active[s]
        delays[s] = d['observer_target_burn_delay_tu']
        if not masks[s].any():
            raise ValueError('Each split must contain a usable reference burn')
    train_rows = np.array([regressor(x['training'][a],y['training'][a],k,L,
        delays['training'][a,k]/periods['training']) for a,k in zip(*np.where(masks['training']))])
    target = y['training'][masks['training']]
    b = dict(history_steps=L,degree=2,configuration=cfg['narx'].copy(),
        maximum_impulse_m_s=cfg['narx']['observer_guidance']['maximum_impulse_m_s'],
        input_scaler=StandardScaler().fit(train_rows),poly=PolynomialFeatures(2,include_bias=False))
    terms = b['poly'].fit_transform(b['input_scaler'].transform(train_rows))
    b['variable_terms'] = np.flatnonzero(terms.std(0)>1e-12)
    b['term_scaler'] = StandardScaler().fit(terms)
    design = b['term_scaler'].transform(terms)[:,b['variable_terms']]
    candidates = []; best = np.inf; selected_models = None; selected_count = None
    for count in (1,2,4,8,12):
        if count >= min(design.shape):
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            b['models'] = [OrthogonalMatchingPursuit(n_nonzero_coefs=count).fit(design,target[:,j]) for j in range(3)]
        p,failures = rollout(b,x['validation'],y['validation'],active['validation'],delays['validation'],periods['validation'])
        m = metrics(p,y['validation'],masks['validation'])
        score = m['vector_rmse_m_s'] if not m['nonfinite_burns'] else None
        candidates.append(dict(terms_per_component=count,validation=m,failures=failures,
                               warnings=[str(w.message) for w in caught]))
        if score is not None and score < best:
            best = score; selected_models = b['models']; selected_count = count
    if selected_models is None:
        raise RuntimeError('All sparse polynomial candidates failed validation')
    b['models'] = selected_models
    feature_names = [f'{name}[k-{lag}]' for lag in range(L-1,-1,-1)
                     for name in ('rx','ry','rz','vx','vy','vz','sin_phase','cos_phase')]
    feature_names += [f'dv{axis}[k-{lag}]' for lag in range(L,0,-1) for axis in 'xyz'] + ['wait_fraction']
    names = b['poly'].get_feature_names_out(feature_names)[b['variable_terms']]
    coefficients = []
    for model in b['models']:
        coefficients.append(dict(intercept_m_s=float(model.intercept_),
            terms=[dict(term=names[i],coefficient_m_s=float(model.coef_[i]))
                   for i in np.flatnonzero(model.coef_)]))
    report = dict(model='sparse quadratic polynomial NARX',history_steps=L,
        timing='Supplied STM schedule; timing and burn/no-burn are NOT predicted',
        selected_max_terms_per_component=selected_count,candidate_terms=terms.shape[1],
        variable_candidate_terms=len(b['variable_terms']),validation_candidates=candidates,
        coefficients_in_standardized_coordinates=coefficients,splits={})
    predictions = {}
    for s in data:
        p,failures = rollout(b,x[s],y[s],active[s],delays[s],periods[s])
        one,_ = rollout(b,x[s],y[s],active[s],delays[s],periods[s],True)
        report['splits'][s] = dict(arcs=len(x[s]),usable_burns=int(masks[s].sum()),
            all_reference_burns=int(active[s].sum()),recursive=metrics(p,y[s],masks[s]),
            reference_history=metrics(one,y[s],masks[s]),failures=failures,
            wait_phase_counts={str(360*f):int(np.sum(masks[s] & np.isclose(delays[s]/periods[s],f)))
                               for f in cfg['narx']['candidate_delay_fractions']})
        predictions[s] = p
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump(b,out/'polynomial_narx.joblib')
    np.savez_compressed(out/'held_out_predictions.npz',reference=y['test'],predicted=predictions['test'],
        mask=masks['test'],day=data['test']['decision_times_tu']*cfg['time_unit_s']/86400,
        wait_phase_deg=delays['test']/periods['test']*360)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return b,report


class PolynomialController:
    """Predict dv at a fixed reference schedule, using own issued-command history."""
    def __init__(self,bundle,cfg,period,schedule):
        self.bundle,self.cfg,self.period,self.schedule = bundle,cfg,period,schedule
        self.x,self.y = [],[]
        self.last_input_max_abs_z = np.nan

    def command(self,nav,t,k,maximum):
        self.x.append(state_input(nav,t,self.period,self.cfg))
        L = self.bundle['history_steps']
        scheduled = bool(self.schedule['maneuver_requested'][k] and self.schedule['predicted_valid'][k]
                         and not self.schedule['maneuver_rejected'][k])
        dv = np.zeros(3); delay = 0.; invoked = False
        if scheduled:
            delay = float(self.schedule['predicted_delay_tu'][k])
            if k<L:
                # Shared startup only; later reference vectors never enter NARX.
                dv = self.schedule['predicted_delta_v_rotating_m_s'][k].copy()
            else:
                invoked = True
                row = regressor(np.asarray(self.x),np.asarray(self.y),k,L,delay/self.period)
                self.last_input_max_abs_z = float(np.max(np.abs(self.bundle['input_scaler'].transform(row[None]))))
                dv = predict(self.bundle,row)[0]
        valid = bool(np.isfinite(dv).all() and np.linalg.norm(dv)<=maximum)
        self.y.append(dv.copy() if valid else np.zeros(3))
        return dict(required=scheduled,valid=valid,burn=dv,delay_tu=delay,invoked=invoked)
