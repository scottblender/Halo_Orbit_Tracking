"""Build the two short, executable course notebooks (no copied output images)."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells):
    result = nb.v4.new_notebook()
    result.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    result.cells = [nb.v4.new_markdown_cell(text.strip()) if kind == 'md'
                    else nb.v4.new_code_cell(text.strip()) for kind, text in cells]
    return result


SETUP = '''
from pathlib import Path
import json
import numpy as np
from IPython.display import display, Markdown
from halo_tracking import stationkeeping as sk
from halo_tracking import stationkeeping_diagnostics as sd
from halo_tracking import stationkeeping_plots as plots
ROOT = Path.cwd()
if not (ROOT/'configs').exists():
    ROOT = ROOT.parent
OUT = ROOT/'artifacts'/'stationkeeping'
plots.style()
def table(headers, rows):
    display(Markdown('| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'+
                     '\\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)))
'''


INTRO = [
('md', r'''
# 1. Identifying halo-orbit station-keeping dynamics

**Question:** Can a phase-dependent state-space model predict the response to a small impulse, and use that prediction to choose its magnitude and orbital phase so that position error stays below **10 km** and velocity error below **0.1 m/s**?

A near-rectilinear halo orbit (**NRHO**) is a repeating path near the Moon. Small departures can grow without corrective velocity changes, called **impulses** or $\Delta v$. We study only the observer's own orbit. Phase is elapsed time through one period, expressed from 0° to 360°; it is not a geometric angle.

The identification plant maps three commanded impulse components $u_k$ [m/s] to six position/velocity departure components $y_k$. The guidance calculation later uses this plant in the reverse direction to choose a burn. These are separate tasks.
'''),
('code', SETUP+'''
orbit, data = sk.prepare_study(ROOT, OUT/'data')
training = data['training']
table(['Split','Independent trajectories','Seed range'],
      [(name,len(rec.seeds),f'{rec.seeds[0]}–{rec.seeds[-1]}') for name,rec in data.items()])
print(f'Uniform sampling: {orbit.dt_s/3600:.3f} h; nominal period: {orbit.T*orbit.cfg["time_unit_s"]/86400:.3f} day')
print('Open-loop random test impulses; independent synthetic full-state navigation measurements.')
'''),
('md', r'''
## Input and output, by direction

$x$ points along the Earth–Moon line, $y$ lies in the orbital plane, and $z$ points out of that plane. Position errors use rotating axes. Velocity errors are **inertial velocity differences resolved on those same axes**, including the frame's transport term. Each displayed output is measured **before** its indexed impulse; $u_k$ affects $y_{k+1}$. The staircase shows impulse samples, not a continuously applied acceleration.

These are simulated identification experiments, not flight telemetry or an optical navigation demonstration. Small independent random commands provide excitation in all three directions. Whole trajectories have separate training, validation and test seeds; test data are reserved for Notebook 2.
'''),
('code', "plots.input_output(orbit, training, OUT/'figures')"),
('md', r'''
## Welch spectra and empirical frequency response

Following Studios 7–8, start with a Hann window, 50% overlap and 32 samples per segment, then repeat with 16 and 64 to expose the resolution/variance trade. Average spectra **within** separate trajectories before forming ratios. Never join endpoints from different experiments.

The full three-input estimate is $\hat H=\hat S_{yu}\hat S_{uu}^{-1}$; the cross-spectrum uses $Y\,U^*$. Pairwise coherence screens weak channels, while input spectral conditioning screens unreliable inversion. Because an orbit can be phase dependent and departures can be nonstationary, these are **finite-record summaries**, not an established global LTI transfer function. Input PSD bands use the conservative non-overlapping segment count; formal output/FRF confidence bands are deferred until stationarity is justified.
'''),
('code', '''
s = orbit.settings
spec = sd.welch_bundle(training.u, training.y[:,:-1], 1/orbit.dt_s,
                       s['welch_length'], s['welch_overlap'])
table(['Length','Resolution [cycle/day]','Overlapping windows','Conservative DOF'],
      [(L, f'{86400/(orbit.dt_s*L):.3f}',
        sd.welch_bundle(training.u,training.y[:,:-1],1/orbit.dt_s,L)['segments'],
        sd.welch_bundle(training.u,training.y[:,:-1],1/orbit.dt_s,L)['dof']) for L in (16,32,64)])
plots.spectra(spec, orbit, OUT/'figures')
plots.frf(spec, OUT/'figures')
'''),
('code', '''
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,2,figsize=(15,5))
for L in (16,32,64):
    q = sd.welch_bundle(training.u,training.y[:,:-1],1/orbit.dt_s,L)
    axs[0].semilogy(q['f'][1:]*86400,q['syy'][1:,0]/86400,label=f'L={L}')
    axs[1].semilogy(q['f'][1:]*86400,abs(q['H'][1:,0,0]),label=f'L={L}')
axs[0].set(xlabel='Frequency [cycle/day]',ylabel='x position PSD [km²/(cycle/day)]')
axs[1].set(xlabel='Frequency [cycle/day]',ylabel='Δv x → x position gain [km/(m/s)]')
for ax in axs:
    ax.grid(); ax.legend()
plots.finish(fig,OUT/'figures','02b_welch_resolution_sensitivity')
'''),
('md', r'''
## What structure do the data support?

Studio 6 compares segment means, variances and autocovariances, then pairs ADF (unit-root null) with KPSS (stationarity null). Here the record is short and periodically varying dynamics can change its statistics. ADF/KPSS are screens with limited power; even agreement does not establish wide-sense stationarity. Shaded segment bands account approximately for correlation and are diagnostic, not simultaneous confidence regions.
'''),
('code', '''
stationarity_rows = [sd.stationarity(training.u[0,:,c],'u '+ 'xyz'[c]) for c in range(3)]
stationarity_rows += [sd.stationarity(training.y[0,:,c],plots.NAMES[c]) for c in range(6)]
table(['Channel','ADF p','KPSS p*','Mean flags / 4','Variance flags / 4','Joint reading'],
      [(r['channel'],f"{r['adf_p']:.3g}",f"{r['kpss_p']:.3g}",r['mean_flags'],r['variance_flags'],r['reading'])
       for r in stationarity_rows])
print('* KPSS values at 0.01/0.10 can be table bounds; exact warnings are saved in stationarity.json.')
(OUT/'stationarity.json').write_text(json.dumps(stationarity_rows,indent=2)+'\\n')
plots.stationarity_panels(training,orbit,OUT/'figures')
'''),
('md', r'''
The next plots examine memory (input–output correlations), excitation rank, coupling across directions, and sensitivity to **orbital phase**. Compare nonlinear propagation with the linear state transition matrix (**STM**) across perturbation sizes: this tests the range over which linearization is useful. It does not prove global linearity. Unlike a damped laboratory structure, an uncontrolled halo orbit need not have stable LTI poles.
'''),
('code', '''
diagnostics = sk.finite_horizon_diagnostics(orbit)
ccf = sd.lag_response(training.u,training.y[:,:-1],16)
plots.structure(orbit,training,diagnostics,ccf,spec,OUT/'figures')
np.savez_compressed(OUT/'structure_diagnostics.npz',**diagnostics)
print('Largest one-orbit Floquet multiplier magnitude:',float(abs(diagnostics['multipliers']).max()))
print('Reference closure [position km; velocity m/s]:',orbit.closure)
table(['Property','Interpretation'],[
    ('Plant','Nonlinear autonomous Earth–Moon CR3BP, with discrete impulses'),
    ('Departure model','Locally linear, periodically time varying about the nominal orbit'),
    ('Input/output','3 commanded impulses → 6 observed state departures; coupled MIMO'),
    ('Measurements','Full-state synthetic navigation; C=I, not an optical observability claim'),
    ('Identification','Open loop; known commands independent of navigation noise'),
    ('Guidance','Closed loop; phase and impulse selected from current estimated state')])
'''),
('md', r'''
## Why station keeping is needed

The nominal orbit is numerically corrected to close before using its STM. Floquet multipliers describe growth over one orbit; a multiplier outside the unit circle indicates local linear instability. The following comparison propagates one small initial error continuously, with **no physical resets** and no burns. The dashed lines mark the required corridor.
'''),
('code', '''
no_burn_day,no_burn_output = sk.uncontrolled(orbit)
plots.geometry(orbit,no_burn_day,no_burn_output,OUT/'figures')
np.savez_compressed(OUT/'uncontrolled.npz',day=no_burn_day,output=no_burn_output)
'''),
('md', r'''
## Working choice for Notebook 2

Start with the six-state, phase-dependent STM grey box and estimate three actuator gains. It retains known cross-axis dynamics while allowing a measurable physical mismatch. The phase-sensitivity, coupling and linearization plots determine whether a frozen/local LTI approximation is credible; the full-record Welch peaks alone cannot select an LTI order.

Possible later comparisons are local phase-conditioned OE/BJ models or a periodic parameter basis (Studio 13). That decision remains open until these diagnostics are reviewed. **NARX is deferred.** See `docs/course_alignment.md` for method provenance and experiment limits.
''')]


MODEL = [
('md', r'''
# 2. Physics-informed grey-box station keeping

Start with the **state transition matrix (STM)**, which propagates a small departure along the nominal orbit. In corridor-scaled coordinates,

$$z_{k+1}=A_k z_k+B_k\operatorname{diag}(g_x,g_y,g_z)u_k+w_k,\qquad y_k=Cz_k+v_k,$$
$$\dot\Phi(t,t_k)=F(t)\Phi(t,t_k),\quad\Phi(t_k,t_k)=I,\quad
A_k=J^{-1}\Phi(t_{k+1},t_k)J,\quad B_k=A_k E_v.$$

$J$ maps scaled physical departures to the dynamics' normalized coordinates; $E_v$ inserts an impulse in m/s into scaled velocity. $C=I$ for the scaled full-state observation (physical plots restore km and m/s). The six states are three position and three velocity errors. The known dynamics set $A_k,B_k$; only the **three actuator gains** $g$ are identified. Without fitted parameters this would be a white-box prediction, rather than identification.

The nonlinear plant is autonomous, but its linearization along a periodic trajectory is **linear time periodic**. This is a local approximation. Neither the impulse decision nor the full closed loop is globally linear. No NARX model is fitted.
'''),
('code', SETUP+'''
orbit,data = sk.restore_study(ROOT,OUT/'data')
fit = sk.fit_greybox(orbit,data['training'])
gains = fit['gains']
table(['Axis','Estimated gain','Approx. standard error'],
      [('xyz'[c],f'{gains[c]:.5f}',f"{fit['gain_standard_errors'][c]:.3g}") for c in range(3)])
print('Whitened sensitivity singular values:',fit['jacobian_singular_values'])
print('PEM optimizer:',fit['optimizer_message'])
(OUT/'greybox_fit.json').write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v
                                               for k,v in fit.items()},indent=2)+'\\n')
'''),
('md', r'''
## Identification and validation

Following Studios 4–5 and 11, minimize covariance-standardized, one-step prediction errors using training trajectories only. A Kalman predictor propagates the declared navigation covariance and a small fixed linearization-error floor; its covariance is independent of the fitted gains. Consequently this criterion is also the Gaussian innovation maximum-likelihood criterion for the stated model. Standard errors are conditional on that model and may be optimistic if it is wrong.

First compare fixed unity gains and fitted gains on validation trajectories. The grey-box structure is the predeclared starting point, not a winner selected against every possible model. Then evaluate the untouched test split once, including **free simulation**, which receives only its initial observation and known impulses.
'''),
('code', '''
scores = {name:sk.prediction_scores(orbit,rec,gains) for name,rec in data.items()}
unity_validation = sk.prediction_scores(orbit,data['validation'],np.ones(3))
rows = []
for label,q in [('Validation: unity physics',unity_validation)]+[(name,q) for name,q in scores.items()]:
    for mode in ('one_step','free_run'):
        rows.append((label,mode,f"{q[mode]['position_rmse_km']:.5g}",f"{q[mode]['velocity_rmse_m_s']:.5g}"))
table(['Split / model','Prediction','3-D position RMSE [km]','3-D velocity RMSE [m/s]'],rows)
(OUT/'prediction_scores.json').write_text(json.dumps(scores,indent=2)+'\\n')
test = data['test']
pred = sk.predict(orbit,test,gains)
free = sk.predict(orbit,test,gains,free_run=True)
plots.prediction(orbit,test,pred,free,OUT/'figures')
np.savez_compressed(OUT/'test_predictions.npz',**pred,free_simulation=free['predicted'])
'''),
('md', r'''
## Check dynamics separately from the noise model

As in Studio 10, compare the empirical input–output frequency summary with the model response driven by the **same held-out commands**. For a periodic model this is a finite-record comparison, not a single LTI $G(j\omega)$. Inspect amplitude and phase as well as time-domain prediction; close agreement in a weakly excited, low-coherence band carries little evidence.
'''),
('code', '''
length = orbit.settings['welch_length']
observed_spec = sd.welch_bundle(test.u,test.y[:,:-1],1/orbit.dt_s,length)
simulation = np.concatenate([test.y[:,:1],free['predicted']],axis=1)
model_spec = sd.welch_bundle(test.u,simulation[:,:-1],1/orbit.dt_s,length)
plots.frf(observed_spec,OUT/'figures',model_spec=model_spec)
'''),
('md', r'''
## Residual whiteness, input independence and stationarity

Use held-out **one-step innovations**, standardized by their time-dependent covariance. Raw prediction errors can have phase-dependent variance; free-simulation errors are not innovations. Plot ACF and causal residual–input CCF, and calculate Ljung–Box tests separately for each trajectory/channel at 12 lags. The fitted model is fixed on test data, so no fitted ARMA degrees of freedom are subtracted. Holm adjustment handles the declared families of channel/trajectory tests and CCF lag tests. Shaded ACF/CCF bands remain nominal pointwise screens.

Also repeat segment and ADF/KPSS checks on innovations. Whiteness assesses the assumed **noise/predictor model**; it does not prove that the physical dynamics or station-keeping policy are correct. Input correlation can reveal unexplained command effects. Passing either check is insufficient for the corridor requirement.
'''),
('code', '''
residual = sd.residual_report(pred['standardized'],test.u,orbit.settings['residual_lags'])
plots.residuals(residual,test,orbit,OUT/'figures')
table(['Test trajectory']+plots.NAMES,
      [(int(seed),*[f'{p:.3g}' for p in row]) for seed,row in zip(test.seeds,residual['ljung_box_holm_p'])])
print('Ljung–Box entries above are Holm-adjusted p-values; reject the family member at p < 0.05.')
print('CCF entries rejected after Holm:',int((residual['ccf_holm_p']<.05).sum()))
print('Standardized innovation mean:',residual['standardized_mean'])
innovation_stationarity = [sd.stationarity(pred['standardized'][a,:,c],f'{seed}: {plots.NAMES[c]}')
                          for a,seed in enumerate(test.seeds) for c in range(6)]
table(['Test innovation','ADF p','KPSS p*','Joint reading'],
      [(r['channel'],f"{r['adf_p']:.3g}",f"{r['kpss_p']:.3g}",r['reading'])
       for r in innovation_stationarity])
(OUT/'innovation_stationarity.json').write_text(json.dumps(innovation_stationarity,indent=2)+'\\n')
np.savez_compressed(OUT/'residual_checks.npz',**{k:v for k,v in residual.items() if isinstance(v,np.ndarray)})
innovation_record = sk.Experiment(test.u,pred['standardized'],pred['standardized'],test.seeds)
plots.stationarity_panels(innovation_record,orbit,OUT/'figures',prefix='11b_innovation_stationarity')
'''),
('md', r'''
## Predict impulse magnitude and phase

At each decision, propagate the estimated departure over the next eight intervals. Consider no burn or one impulse now, one interval later, or two intervals later. For each time, solve a weighted path least-squares correction, check the candidate with nonlinear propagation, and prefer small impulses that meet a conservative planning margin. The schedule is **chosen by the model**, not supplied from a reference controller. A delayed burn remains pending until execution; no second burn is scheduled while it is pending.

Run the fitted controller against the nonlinear plant with independent noisy navigation measurements. Compare with an uncontrolled plant from the same initial departure. Evaluate **3-D error norms**, including both sides of each impulse and eight samples per coast interval. This finite grid is not a continuous-time guarantee. Constraints can be infeasible for a single impulse, so report observed violations rather than assuming success. The primary trial seed is fixed before seeing its result.
'''),
('code', '''
trial = sk.stationkeeping_trial(orbit,gains)
no_burn_day,no_burn_output = sk.uncontrolled(orbit)
plots.controlled(orbit,trial,no_burn_day,no_burn_output,OUT/'figures')
summary = trial['summary']
display(Markdown('**Corridor result:** '+('all evaluated samples inside both limits.'
                if summary['all_evaluated_samples_inside'] else 'the controller did not satisfy both limits at every evaluated sample.')))
table(['Metric','Observed value'],list(summary.items()))
np.savez_compressed(OUT/'closed_loop.npz',**{k:v for k,v in trial.items() if k!='summary'})
(OUT/'closed_loop_summary.json').write_text(json.dumps(summary,indent=2)+'\\n')
'''),
('md', r'''
## What this establishes, and what comes next

The experiment separates identifying local orbit response from planning a correction. The saved gain fit, held-out predictions, residual diagnostics, commanded burn phases and nonlinear corridor histories make those claims independently checkable. The simulation uses the same CR3BP physics with a deliberately hidden actuator mismatch; it is an identification demonstration, not flight qualification or evidence of robustness to missing forces.

Review the phase/linearity plots and residuals before choosing the next data-driven model. A phase-local OE/BJ or a periodic basis model may be useful; NARX remains deferred. A stronger performance claim needs multiple initial errors/phases, independent disturbance models, realistic navigation, and denser or event-based constraint checks. `docs/course_alignment.md` maps the methods to the uploaded studios.
''')]


if __name__ == '__main__':
    for name,cells in [('01_stationkeeping_identification.ipynb',INTRO),
                       ('02_stm_greybox_stationkeeping.ipynb',MODEL)]:
        nb.write(notebook(cells),ROOT/'notebooks'/name)
        print(name)
