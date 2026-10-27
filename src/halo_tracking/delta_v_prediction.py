"""Three-output halo-observer delta-v prediction and data diagnostics.

Uses the existing STM command dataset; these labels target halo departure,
not a direct Gateway tracking objective. No burn classifier or timing output
is learned here. Existing controller experiments remain available separately.
"""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from narxpy import NARX
from .neural_narx import exogenous, sequence
from .two_stage_narx import _load, _features


def delta_v_targets(data):
    """Zero means known no-command; rejected required labels stay unknown."""
    required = data['observer_maneuver_required'].astype(bool)
    valid = data['observer_maneuver_valid'].astype(bool)
    burn = data['observer_target_delta_v_rotating_m_s']
    known = ~required | (valid & np.isfinite(burn).all(-1))
    active = required & known
    y = np.zeros_like(burn)
    y[active] = burn[active]
    return y, known, active


def usable_rows(x, known, history):
    mask = np.zeros(known.shape, dtype=bool)
    for k in range(history,x.shape[1]):
        mask[:,k] = (known[:,k-history:k+1].all(-1)
                     & np.isfinite(x[:,k-history+1:k+1]).all((1,2)))
    return mask


def prediction_metrics(prediction, target, mask, active):
    error = prediction-target
    burn = mask & active
    rest = mask & ~active
    def vector_rmse(rows):
        return float(np.sqrt(np.mean(np.sum(error[rows]**2,-1)))) if rows.any() else None
    return dict(decisions=int(mask.sum()),burn_examples=int(burn.sum()),
        vector_rmse_m_s=vector_rmse(mask),burn_vector_rmse_m_s=vector_rmse(burn),
        no_burn_vector_rmse_m_s=vector_rmse(rest),
        component_rmse_m_s=np.sqrt(np.mean(error[mask]**2,axis=0)).tolist(),
        bias_m_s=error[mask].mean(0).tolist())


def baseline_predictions(bundle,data):
    """Use unchanged gradient-boosted execute classifier and dv regressors.

    Score raw predicted vectors: impulse-cap/timing rejection from the old
    actuator is not substituted for a vector prediction in this regression test.
    """
    features,usable = _features(data,'observer')
    result = np.full((len(features),3),np.nan)
    rows = usable & np.isfinite(features).all(-1)
    result[rows] = 0.
    active = np.flatnonzero(rows)[bundle['classifier'].predict(features[rows]).astype(bool)]
    if len(active):
        result[active] = np.column_stack([r.predict(features[active]) for r in bundle['regressors']])
    return result.reshape(*data['observer_maneuver_required'].shape,3)


def fit_delta_v_predictor(dataset_dir,cfg,output_dir,baseline,epochs=600,hidden=64):
    """Fit on training trajectories; select on validation; test after fitting."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg['narx']['seed'])
    torch.use_deterministic_algorithms(True)
    data = {s:_load(Path(dataset_dir)/(s+'.npz')) for s in ('training','validation','test')}
    x = {s:exogenous(d) for s,d in data.items()}
    labels = {s:delta_v_targets(d) for s,d in data.items()}
    L = int(cfg['narx']['history_steps'])
    masks = {s:usable_rows(x[s],labels[s][1],L) for s in data}
    if not all(m.any() for m in masks.values()):
        raise ValueError('Every split needs known targets and valid causal histories')
    y = {s:v[0] for s,v in labels.items()}
    # Neither validation nor test participates in scaler fitting.
    input_rows = np.isfinite(x['training']).all(-1)
    xm = x['training'][input_rows].mean(0)
    xs = np.maximum(x['training'][input_rows].std(0),1e-6)
    ym = y['training'][labels['training'][1]].mean(0)
    ys = np.maximum(y['training'][labels['training'][1]].std(0),1e-3)
    network = NARX(L,L,11,3,hidden,'Identity')
    b = dict(network=network,history_steps=L,x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys,hidden=hidden)
    opt = torch.optim.Adam(network.parameters(),lr=.003,weight_decay=1e-4)
    tm = torch.tensor(masks['training']); vm = torch.tensor(masks['validation'])
    loss_history,best,selected,state = [],float('inf'),None,None
    for epoch in range(epochs):
        p,t = sequence(b,x['training'],y['training'],True)
        loss = (p[tm]-t[tm]).square().mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
        opt.step()
        with torch.no_grad():
            pv,tv = sequence(b,x['validation'],y['validation'],True)
            val = float((pv[vm]-tv[vm]).square().mean())
        loss_history.append([epoch+1,float(loss.detach()),val])
        if val < best:
            best,selected,state = val,epoch+1,copy.deepcopy(network.state_dict())
    network.load_state_dict(state); network.eval()
    report = dict(output='rotating-frame halo observer delta-v vector [m/s]',
        label_objective='STM halo-reference departure correction, not direct Gateway tracking optimization',
        epochs_trained=epochs,selected_epoch=selected,history_steps=L,
        parameters=sum(p.numel() for p in network.parameters()),splits={})
    predictions = {}
    with torch.no_grad():
        for split in ('validation','test'):
            predictions[split] = {}
            for mode,teacher in [('NARX one-step',True),('NARX recursive',False)]:
                pred,_ = sequence(b,x[split],y[split],teacher)
                predictions[split][mode] = pred.numpy()*ys+ym
            predictions[split]['Gradient-boosted'] = baseline_predictions(baseline,data[split])
            predictions[split]['Zero delta-v'] = np.zeros_like(y[split])
            # Causal persistence includes only past known teacher commands.
            predictions[split]['Previous delta-v'] = np.concatenate((np.zeros_like(y[split][:,:1]),y[split][:,:-1]),1)
            report['splits'][split] = {name:prediction_metrics(p,y[split],masks[split],labels[split][2])
                for name,p in predictions[split].items()}
    diagnostics = {}
    for split,d in data.items():
        mask,active = masks[split],labels[split][2]
        bad_tracking = ((d['gateway_tracking_position_error_km']>.1)
                        | (d['gateway_tracking_velocity_error_m_s']>.01)) & mask
        diagnostics[split] = dict(arcs=len(x[split]),decisions=int(mask.sum()),
            burn_examples=int((mask & active).sum()),no_burn_fraction=float((mask & ~active).sum()/mask.sum()),
            burn_examples_per_arc=(mask & active).sum(-1).tolist(),
            tracking_limit_exceeded_decisions=int(bad_tracking.sum()),
            tracking_limit_exceeded_without_command=int((bad_tracking & ~active).sum()),
            position_limit_km=.1,velocity_limit_m_s=.01,
            unknown_requested_targets=int((~labels[split][1]).sum()))
    report['data_diagnostics'] = diagnostics
    out = Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    np.save(out/'training_loss.npy',np.asarray(loss_history))
    np.savez_compressed(out/'held_out_predictions.npz',target=y['test'],mask=masks['test'],
        burn=labels['test'][2],one_step=predictions['test']['NARX one-step'],
        recursive=predictions['test']['NARX recursive'],baseline=predictions['test']['Gradient-boosted'],
        day=data['test']['decision_times_tu']*cfg['time_unit_s']/86400)
    torch.save(dict(state_dict=network.state_dict(),history_steps=L,hidden=hidden,
        x_mean=xm.tolist(),x_scale=xs.tolist(),y_mean=ym.tolist(),y_scale=ys.tolist()),out/'delta_v_narx.pt')
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return b,report,predictions,data,masks
