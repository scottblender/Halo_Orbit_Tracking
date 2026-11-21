"""Build concise LSO-based course notebooks with live purpose/result captions."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]


def notebook(cells):
    n=nb.v4.new_notebook()
    n.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    n.cells=[nb.v4.new_markdown_cell(t.strip()) if k=='md' else nb.v4.new_code_cell(t.strip()) for k,t in cells]
    return n


SETUP = '''
from pathlib import Path
import json
import numpy as np
from IPython.display import display, Markdown
from halo_tracking import stationkeeping as physics
from halo_tracking import lso_stationkeeping as sk
from halo_tracking import stationkeeping_diagnostics as sd
from halo_tracking import lso_diagnostics as ld
from halo_tracking import lso_report_plots as plots
ROOT=Path.cwd()
if not (ROOT/'configs').exists(): ROOT=ROOT.parent
OUT=ROOT/'artifacts'/'stationkeeping'
OUT.mkdir(parents=True,exist_ok=True)
plots.style()
def table(headers,rows):
    display(Markdown('| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'+
                     '\\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)))
'''


INTRO = [
('md', r'''
# 1. LSO data and NRHO station-keeping identification

**Question:** Can an LSO-informed, phase-dependent model choose the impulse magnitude and orbital phase needed to keep a spacecraft within **10 km** in position and **0.1 m/s** in velocity of its nominal NRHO?

An NRHO is a repeating path near the Moon. An impulse is a short velocity change that corrects departure from that path. Three **lunar surface observatories (LSOs)** measure line-of-sight angles, not position or velocity directly. Their noisy angles and actual visibility gaps feed a state estimator. Simulation truth is reserved for generating sensors and evaluating accuracy.

Inputs are commanded impulses in three directions. Reconstructed outputs are position and inertial-velocity departures resolved along rotating Earth–Moon axes. Phase means time through the orbit in degrees, not a geometric angle. Measurements at epoch $k$ precede impulse $u_k$, which affects the next state.
'''),
('code', SETUP+'''
orbit,data=sk.prepare_study(ROOT,OUT/'data')
training=data['training']
table(['Split','Independent trajectories','Seed range'],[(n,len(r.seeds),f'{r.seeds[0]}–{r.seeds[-1]}') for n,r in data.items()])
print(f'LSO angle noise: {orbit.cfg["measurement"]["angle_sigma_arcsec"]:g} arcsec. Uniform cadence: {orbit.dt_s/3600:.2f} hour.')
plots.inputs_outputs(orbit,training,OUT/'figures')
'''),
('md', r'''
## What is actually measured?

The LSO channels are inertial-axis angles $\alpha$ and $\delta$. Elevation, range and configured Sun/Earth exclusions determine availability. Missing angles remain blank, while a finite state estimate during an outage is a model prediction, not a new measurement. The block diagram separates this measurement path from the station-keeping feedback.
'''),
('code', '''
plots.lso_measurements(orbit,training,OUT/'figures')
plots.diagram(orbit,OUT/'figures')
'''),
('md', r'''
## Welch spectra and empirical response

Following Studios 7–8, use Hann windows and 50% overlap, averaging within separate trajectories. The full MIMO estimate is $\hat H=\hat S_{yu}\hat S_{uu}^{-1}$ with cross-spectrum $Y U^*$. The displayed channels are its three direct position responses. Because $y$ is an **LSO-derived state estimate**, its response includes estimator dynamics and is not the physical plant transfer function alone.

Compare window lengths before interpreting peaks: quarter-orbit windows cannot resolve the orbital fundamental, while full-orbit windows leave one window per trajectory. These are finite-record summaries. A stationary output spectrum or global LTI transfer function is not assumed.
'''),
('code', '''
s=orbit.settings
spec=sd.welch_bundle(training.u,training.y[:,:-1],1/orbit.dt_s,s['welch_length'])
table(['Window length','Resolution [cycle/day]','Averaged windows'],[(L,f'{86400/(orbit.dt_s*L):.3g}',sd.welch_bundle(training.u,training.y[:,:-1],1/orbit.dt_s,L)['segments']) for L in (16,32,64,128)])
plots.spectra(orbit,spec,OUT/'figures')
plots.frf(spec,OUT/'figures')
'''),
('md', r'''
## Stationarity and model structure

Studio 6 compares segment means/variances and pairs ADF (unit-root null) with KPSS (level-stationarity null). Here the reconstructed states include estimator dynamics. Angular gaps are never filled to manufacture measurements. The phase/linearity/Floquet plots then test whether a local time-invariant approximation is credible and whether perturbations grow over an orbit.
'''),
('code', '''
names=['r x','r y','r z','v x','v y','v z']
stationarity_rows=[sd.stationarity(training.y[0,:,c],names[c]) for c in range(6)]
table(['Channel','ADF p','KPSS p*','Joint reading'],[(r['channel'],f"{r['adf_p']:.3g}",f"{r['kpss_p']:.3g}",r['reading']) for r in stationarity_rows])
print('* KPSS table-bound warnings are retained in the saved report.')
(OUT/'stationarity.json').write_text(json.dumps(stationarity_rows,indent=2)+'\\n')
plots.stationarity(orbit,training,stationarity_rows,OUT/'figures')
diagnostics=physics.finite_horizon_diagnostics(orbit)
plots.structure(diagnostics,OUT/'figures')
np.savez_compressed(OUT/'structure_diagnostics.npz',**diagnostics)
'''),
('md', r'''
## Longer propagation shows uncontrolled growth

Propagate the same small initial departure continuously for **up to 24 orbits**, stopping at a declared 20,000 km departure cutoff or integration failure. Only the reference repeats. The spacecraft is never reset. Orthographic orbit projections replace the overlapping 3-D axes, alongside position and velocity growth panels.
'''),
('code', '''
no_day,no_y,no_summary=sk.uncontrolled(orbit)
plots.geometry_growth(orbit,no_day,no_y,no_summary,OUT/'figures')
np.savez_compressed(OUT/'uncontrolled.npz',day=no_day,output=no_y)
(OUT/'uncontrolled_summary.json').write_text(json.dumps(no_summary,indent=2)+'\\n')
'''),
('md', r'''
## What does this analysis establish?

From this analysis, the NRHO station-keeping system requires a **nonlinear, time-varying model**. The gravitational forces and LSO line-of-sight angles depend nonlinearly on the spacecraft state. The departure dynamics and measurement geometry also change with orbital phase and LSO visibility.

The physical CR3BP dynamics are autonomous in rotating coordinates, meaning the gravitational model does not explicitly depend on time. However, the STM is evaluated along the moving nominal orbit, so its coefficients change with orbital phase. Thus, small departures can be described by a **linear time-periodic STM**, with further changes in the measurement equation when LSOs enter or leave view.

Thus, the STM provides a starting point for local grey-box identification. A short-record stationarity test does not remove the measured phase dependence or establish global LTI behavior. NARX remains deferred.
'''),
('code', '''
response=np.linalg.norm(diagnostics['impulse_sensitivity'],axis=(1,2))
radius=float(abs(diagnostics['multipliers']).max())
support=sum(r['adf_p']<.05 and r['kpss_p']>=.05 for r in stationarity_rows)
display(Markdown(f'**Observed evidence:** horizon sensitivity varies by a factor of {response.max()/response.min():.2g} across phase, the largest one-orbit multiplier is {radius:.3g}, and {support}/6 reconstructed state channels give the ADF/KPSS combination supporting level stationarity.'))
display(Markdown('**Stability conclusion:** '+('the multiplier outside the unit circle identifies a locally unstable mode, and the extended uncontrolled trajectory shows its finite-time consequences.' if radius>1.001 else 'these multiplier calculations do not robustly resolve instability.')))
''')]


MODEL = [
('md', r'''
# 2. An LSO-informed STM grey box

The six states are position and velocity departure from the nominal NRHO. In corridor-scaled coordinates,

$$e_{k+1}=A_k e_k+B_k\operatorname{diag}(g)u_k+w_k,\qquad z_{j,k}=h_j(t_k,x_{\rm nom}(t_k))+C_{j,k}e_k+v_{j,k},$$
$$\dot\Phi=F(t)\Phi,\quad A_k=J^{-1}\Phi(t_{k+1},t_k)J,\quad C_{j,k}=\left.\frac{\partial h_j}{\partial x}\right|_{x_{\rm nom}(t_k)}J.$$

$z_{j,k}$ contains the two angles from visible LSO $j$, so **$C_{j,k}$ is not the identity**. Known physics fixes the STM. Training estimates three actuator gains $g$. This is a locally linear, phase-dependent model of the nonlinear orbit/angle system.
'''),
('code', SETUP+'''
orbit,data=sk.restore_study(ROOT,OUT/'data')
fit=sk.fit_greybox(orbit,data['training']);gains=fit['gains']
table(['Axis','Estimated gain','Conditional standard error'],[('xyz'[j],f'{gains[j]:.4g}',f"{fit['gain_standard_errors'][j]:.3g}") for j in range(3)])
print('Angular-innovation sensitivity singular values:',fit['jacobian_singular_values'])
(OUT/'greybox_fit.json').write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()},indent=2)+'\\n')
'''),
('md', r'''
## Identification and held-out validation

Following Studios 5 and 11, minimize covariance-standardized **raw angular innovations** on training trajectories, updating only when an LSO is visible. No generated full-state observations or saved truth enter the fit. Filtered state histories are correlated outputs, so they support exploratory plots and guidance, not independent-measurement regression.

Validation compares unity and fitted gains. Independent test trajectories assess angular prediction and navigation accuracy. Free propagation uses its initial angular update and then the known commands, with no later measurements. Truth is used only to evaluate state errors.
'''),
('code', '''
scores={name:sk.prediction_scores(orbit,rec,gains) for name,rec in data.items()}
unity=sk.prediction_scores(orbit,data['validation'],np.ones(3))
table(['Dataset / model','Angular RMSE [arcsec]','Navigation position RMSE [km]','Navigation velocity RMSE [m/s]'],[(name,f"{q['angular_innovation_rmse_arcsec']:.3g}",f"{q['navigation']['position_rmse_km']:.3g}",f"{q['navigation']['velocity_rmse_m_s']:.3g}") for name,q in [('Validation: unity gains',unity)]+list(scores.items())])
(OUT/'prediction_scores.json').write_text(json.dumps(scores,indent=2)+'\\n')
test=data['test'];pred=sk.predict(orbit,test,gains);free=sk.predict(orbit,test,gains,free_run=True)
plots.validation(orbit,test,pred,free,OUT/'figures')
np.savez_compressed(OUT/'test_predictions.npz',**pred,free_simulation=free['navigation'])
'''),
('md', r'''
## Do the residuals support the predictor?

Use standardized angular innovations with actual time lags and visibility gaps intact. Ljung–Box uses eligible contiguous runs with Holm adjustment. A mask-conditioned Monte Carlo maximum-CCF test covers the input/channel/lag family. No gaps or independent trajectories are concatenated, and no fitted ARMA degrees of freedom are subtracted for the fixed model on test data.

ADF/KPSS also uses contiguous observed runs, reporting insufficient support explicitly. Residual whiteness checks whether the assumed noise model is adequate. Residual–input correlation checks whether the commanded impulses remain related to the prediction error. Passing these checks does not guarantee that the orbit model or the station-keeping controller is accurate.
'''),
('code', '''
residual=ld.residual_report(pred['standardized'],test.u,orbit.settings['residual_lags'])
plots.residuals(residual,OUT/'figures')
stationarity_rows=ld.stationarity_rows(pred['standardized'])
def fmt(p): return 'insufficient data' if p is None else f'{p:.3g}'
table(['Angular channel','Contiguous samples','ADF p','KPSS p','Joint reading'],[(r['channel'],r['contiguous_samples'],fmt(r['adf_p']),fmt(r['kpss_p']),r['reading']) for r in stationarity_rows])
(OUT/'innovation_stationarity.json').write_text(json.dumps(stationarity_rows,indent=2)+'\\n')
np.savez_compressed(OUT/'residual_checks.npz',**{k:v for k,v in residual.items() if isinstance(v,np.ndarray)})
(OUT/'optical_residual_summary.json').write_text(json.dumps({k:v for k,v in residual.items() if not isinstance(v,np.ndarray)},indent=2)+'\\n')
'''),
('md', r'''
## Does LSO-guided station keeping meet the limits?

Guidance selects no burn or an impulse now/one/two intervals later using only the LSO-derived departure estimate, without truth feedback or a supplied burn schedule. The estimator coasts through outages. Candidate burns are checked by nonlinear propagation. Both sides of impulses and coast substeps are evaluated against the **3-D norm** limits.

This single predeclared trial establishes sampled performance, not a continuous-time guarantee or population success rate. A failure can arise from navigation, prediction or the single-impulse planner. This plot alone cannot assign it to a model class.
'''),
('code', '''
trial=sk.stationkeeping_trial(orbit,gains)
if (OUT/'uncontrolled.npz').exists():
    with np.load(OUT/'uncontrolled.npz') as z: no_day,no_y=z['day'],z['output']
else: no_day,no_y,_=sk.uncontrolled(orbit)
plots.controlled(orbit,trial,no_day,no_y,OUT/'figures')
summary=trial['summary']
table(['Metric','Observed value'],[(k,summary[k]) for k in ('completed','last_day','maximum_position_km','maximum_velocity_m_s','all_evaluated_samples_inside','first_sampled_exit_day','burns','total_commanded_delta_v_m_s','visible_epoch_fraction')])
np.savez_compressed(OUT/'closed_loop.npz',**{k:v for k,v in trial.items() if k!='summary'})
(OUT/'closed_loop_summary.json').write_text(json.dumps(summary,indent=2)+'\\n')
rejects=int(np.nansum(residual['ljung_box_holm_p']<.05))
display(Markdown(f'**Noise-model conclusion:** {rejects}/{residual["eligible_whiteness_tests"]} eligible whiteness tests reject after adjustment and the global residual–input p-value is {residual["ccf_global_p"]:.3g}, so these tests '+('flag predictor/noise inadequacy.' if rejects or residual['ccf_global_p']<.05 else 'do not reject the tested assumptions at 5%, without proving the model correct.')))
display(Markdown('**Station-keeping conclusion:** '+('both required norms remain inside their limits at all evaluated samples of the completed run.' if summary['completed'] and summary['all_evaluated_samples_inside'] else f'the LSO-guided baseline does not establish the full corridor requirement, with observed peaks of {summary["maximum_position_km"]:.3g} km and {summary["maximum_velocity_m_s"]:.3g} m/s.')))
'''),
('md', r'''
## Next model discussion

The physical basis and measured phase dependence support a periodic STM starting model, not a globally LTI one. Review LSO availability, gain uncertainty, raw angular residuals and navigation error before selecting a local OE/BJ model or periodic parameter basis (Studio 13). NARX remains deferred. Improving navigation or the burn planner may matter more than increasing black-box complexity.
''')]


if __name__ == '__main__':
    for name,cells in [('01_stationkeeping_identification.ipynb',INTRO),('02_stm_greybox_stationkeeping.ipynb',MODEL)]:
        nb.write(notebook(cells),ROOT/'notebooks'/name)
        print(name)
