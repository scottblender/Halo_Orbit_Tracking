"""Neural command NARX using narxpy with causal online delay buffers."""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from narxpy import NARX
from .two_stage_narx import _load, _timing_classes


def exogenous(data):
    nav = data['observer_navigation_error_km_m_s']
    phase = np.broadcast_to(data['decision_phase_rad'], nav.shape[:2])
    return np.concatenate((nav, np.sin(phase)[..., None], np.cos(phase)[..., None],
        data['gateway_visible'][..., None], data['lso_visible'].sum(-1)[..., None],
        np.log10(np.maximum(data['position_covariance_trace_normalized'][..., :1], 1e-30))), -1)


def targets(data, cfg):
    """[execute, dv_xyz, delay one-hot]; pending/invalid/no-burn = zero.

    Feedback is the previous issued command, distinct from an executed burn
    which can occur later. Nonfinite rejected STM labels never enter training.
    """
    s = cfg['narx']
    period = data['decision_times_tu'][-1] / s['arc_duration_periods']
    dv = data['observer_target_delta_v_rotating_m_s']
    timing = _timing_classes(data['observer_target_burn_delay_tu'].ravel(),
        s['candidate_delay_fractions'], period).reshape(dv.shape[:2])
    active = (data['observer_maneuver_required'] & data['observer_maneuver_valid']
              & (timing >= 0) & np.isfinite(dv).all(-1))
    y = np.zeros((*active.shape, 4 + len(s['candidate_delay_fractions'])))
    y[..., 0] = active
    y[..., 1:4][active] = dv[active]
    a, b = np.where(active)
    y[a, b, 4 + timing[active]] = 1.
    return y


def network_step(model, xx, yy):
    """Exactly the library one-step map, for online simulation."""
    return model.act2(model.hl2(model.act1(model.hl1(
        torch.cat((xx.flatten(1), yy.flatten(1)), 1)))))


def sequence(bundle, x, y, teacher):
    model, L = bundle['network'], bundle['history_steps']
    xs = torch.tensor((x-bundle['x_mean'])/bundle['x_scale'], dtype=torch.float32)
    ys = torch.tensor((y-bundle['y_mean'])/bundle['y_scale'], dtype=torch.float32)
    # narxpy uses x[t-L:t]; shift so this is navigation[t-L+1:t+1].
    shifted = torch.cat((xs[:, 1:], xs[:, -1:]), 1)
    if teacher:
        return model(shifted, mode='open', y=ys), ys
    initial = torch.tensor(-bundle['y_mean']/bundle['y_scale'], dtype=torch.float32)
    bootstrap = initial.expand(xs.shape[0], xs.shape[1], -1).clone()
    return model(shifted, mode='close', y=bootstrap, bootstrap=L), ys


def fit_neural_policy(dataset_dir, cfg, output_dir, epochs=600, hidden=64):
    """Train-only scalers; validation checkpoint selection; held-out test."""
    torch.set_num_threads(1)
    torch.manual_seed(cfg['narx']['seed'])
    torch.use_deterministic_algorithms(True)
    splits = {s: _load(Path(dataset_dir)/(s+'.npz')) for s in ('training','validation','test')}
    x = {s: exogenous(d) for s,d in splits.items()}
    y = {s: targets(d,cfg) for s,d in splits.items()}
    L = int(cfg['narx']['history_steps'])
    if x['training'].shape[1] <= L or not y['training'][:, L:, 0].any():
        raise ValueError('Need usable histories and feasible training commands')
    xm, xs = x['training'].mean((0,1)), np.maximum(x['training'].std((0,1)),1e-6)
    ym, ys = y['training'].mean((0,1)), np.maximum(y['training'].std((0,1)),.1)
    model = NARX(d_i=L,d_o=L,d_x=11,d_y=y['training'].shape[-1],d_hl=hidden,act_func='Identity')
    bundle = dict(network=model,history_steps=L,x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys,
        backend='narxpy 0.1.0',hidden=hidden,
        candidate_delay_fractions=cfg['narx']['candidate_delay_fractions'],
        fitted_training_decisions=int(y['training'][:,L:,0].size),
        fitted_feasible_burns=int(y['training'][:,L:,0].sum()),training_only=True,
        configuration=copy.deepcopy(cfg['narx']))
    optimizer = torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-4)
    history, best, state = [], float('inf'), None
    for epoch in range(epochs):
        model.train()
        pred,true = sequence(bundle,x['training'],y['training'],True)
        loss = (pred[:,L:]-true[:,L:]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pv,tv = sequence(bundle,x['validation'],y['validation'],True)
            val = float((pv[:,L:]-tv[:,L:]).square().mean())
        history.append([epoch+1,float(loss.detach()),val])
        if val < best:
            best,state = val,copy.deepcopy(model.state_dict())
            bundle['selected_epoch'] = epoch+1
    model.load_state_dict(state)
    model.eval()
    report = dict(backend=bundle['backend'],selected_epoch=bundle['selected_epoch'],epochs_trained=epochs,
        training_decisions=bundle['fitted_training_decisions'],feasible_training_burns=bundle['fitted_feasible_burns'],splits={})
    with torch.no_grad():
        for split in ('validation','test'):
            report['splits'][split] = {}
            for mode,teacher in [('teacher_forced',True),('recursive',False)]:
                pred,_ = sequence(bundle,x[split],y[split],teacher)
                p = pred.numpy()[:,L:]*ys+ym
                truth = y[split][:,L:]
                active,gate = truth[...,0]>.5,p[...,0]>.5
                tp = int((gate & active).sum())
                precision,recall = tp/max(1,int(gate.sum())),tp/max(1,int(active.sum()))
                report['splits'][split][mode] = dict(decisions=int(active.size),feasible_burns=int(active.sum()),
                    precision=precision,recall=recall,f1=2*precision*recall/max(1e-12,precision+recall),
                    all_command_rmse_m_s=float(np.sqrt(np.mean((p[...,1:4]-truth[...,1:4])**2))),
                    feasible_burn_rmse_m_s=float(np.sqrt(np.mean((p[...,1:4][active]-truth[...,1:4][active])**2))) if active.any() else None,
                    delay_accuracy=float(np.mean(p[...,4:][active].argmax(-1)==truth[...,4:][active].argmax(-1))) if active.any() else None)
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'training_loss.npy',np.asarray(history))
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(state_dict=model.state_dict(),metadata={k:v.tolist() if isinstance(v,np.ndarray) else v
        for k,v in bundle.items() if k!='network'}),out/'neural_narx.pt')
    return bundle,report


def load_neural_policy(path):
    """Load the portable tensor/plain-data checkpoint without pickled modules."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    b = saved['metadata']
    for name in ('x_mean','x_scale','y_mean','y_scale'):
        b[name] = np.asarray(b[name],dtype=float)
    L = b['history_steps']
    b['network'] = NARX(L,L,11,len(b['y_mean']),b['hidden'],'Identity')
    b['network'].load_state_dict(saved['state_dict'])
    b['network'].eval()
    return b


class NeuralController:
    """Independent per-arc state, initialized with L zero commands."""
    def __init__(self,bundle):
        self.bundle = bundle
        self.inputs,self.outputs = [],[]

    def command(self,current,period,maximum,pending=False):
        b,L = self.bundle,self.bundle['history_steps']
        self.inputs.append(np.asarray(current))
        if len(self.outputs)<L:
            prediction = np.zeros(len(b['y_mean']))
        else:
            xx = (np.asarray(self.inputs[-L:])-b['x_mean'])/b['x_scale']
            yy = (np.asarray(self.outputs[-L:])-b['y_mean'])/b['y_scale']
            with torch.no_grad():
                prediction = network_step(b['network'],torch.tensor(xx[None],dtype=torch.float32),
                    torch.tensor(yy[None],dtype=torch.float32)).numpy()[0]*b['y_scale']+b['y_mean']
        invoked = len(self.outputs)>=L and not pending
        if pending:
            prediction[:] = 0.
        self.outputs.append(prediction.copy())
        required = bool(invoked and prediction[0]>.5)
        dv = prediction[1:4]
        valid = bool(np.isfinite(prediction).all() and np.linalg.norm(dv)<=maximum)
        delay = b['candidate_delay_fractions'][int(np.argmax(prediction[4:]))]*period
        return dict(required=required,valid=valid,burn=dv.copy() if required else np.zeros(3),
                    delay_tu=float(delay) if required else 0.,invoked=invoked)
