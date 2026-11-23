"""Build concise LSO-based course notebooks with live purpose/result captions."""
import argparse
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

The commanded input is the impulse $u_k$ in m/s. The physical state output is the departure $e_k=[\delta r_k,\delta v_k]$. The measured output used for identification is the pair of inertial-axis angles $z_{j,k}=[\alpha,\delta]$ from each visible LSO. For a visible site, $z_{j,k}=h_j(t_k,x_k)+v_{j,k}$, where $h_j$ is the ideal nonlinear line-of-sight measurement.

The simulator injects $v_{j,k}$ additively at the sensor output, after computing $h_j$ and before the visibility mask. Each angle component has independent zero-mean Gaussian noise with $\sigma_\theta=1$ arcsec, so $R=\sigma_\theta^2 I_2$ and the assumed measurement noise is white across epochs and sites. This is a synthetic sensor-noise assumption. There are no real LSO calibration residuals here to establish that an actual instrument has white or Gaussian noise. Bias and colored noise are not modeled. The innovation tests check consistency of this assumed model with the simulation, not real sensor hardware. Missing angles are caused by visibility, not by the noise model.
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

Propagate the same small initial departure continuously for **up to 24 orbits**, stopping at a declared 20,000 km departure cutoff or integration failure. Only the reference repeats. The spacecraft is never reset. A 3D Moon-centred orbit view shows the reference path, Moon and LSO locations. Its legend sits outside the plotting axes, alongside position and velocity growth panels.
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
# 2. Comparing global and periodic STM grey boxes

The six states are position and velocity departure from the nominal NRHO. In corridor-scaled coordinates,

$$e_{k+1}=A_k e_k+B_k\operatorname{diag}(g(\phi_k))u_k+w_k,\qquad z_{j,k}=h_j(t_k,x_{\rm nom}(t_k))+C_{j,k}e_k+v_{j,k},$$
$$\dot\Phi=F(t)\Phi,\quad A_k=J^{-1}\Phi(t_{k+1},t_k)J,\quad C_{j,k}=\left.\frac{\partial h_j}{\partial x}\right|_{x_{\rm nom}(t_k)}J.$$

$z_{j,k}$ contains the two angles from visible LSO $j$, so **$C_{j,k}$ is not the identity**. The measurement noise enters additively after the ideal angle map: $v_{j,k}\sim\mathcal{N}(0,R)$ with $R=\sigma_\theta^2 I_2$ and $\sigma_\theta=1$ arcsec in the synthetic sensor generator. Draws are independent across angle components, sites and epochs, which is a white-noise assumption. No real sensor calibration data are used to justify white or Gaussian noise, and bias or colored noise are not modeled. The innovation tests check this assumption against the synthetic data. Known physics fixes the repeatable $A_k$ and $B_k$ along the nominal orbit. The global baseline fits one set of three mean gains. The periodic fit replaces $g$ with $g(\phi_k)$, an interpolation of overlapping phase windows. The last window blends into the first. The simulator uses constant physical gains, so any fitted phase variation represents effective model correction rather than changing actuator hardware.
'''),
('code', SETUP+'''
orbit,data=sk.restore_study(ROOT,OUT/'data')
global_fit=sk.fit_greybox(orbit,data['training'])
fit=sk.fit_periodic_greybox(orbit,data['training'],
    windows=orbit.settings['periodic_gain_windows'],
    smoothing=orbit.settings['periodic_gain_smoothing'])
gains=fit['gains']
models={'Global gains':global_fit['gains'],'Periodic gains':gains}
table(['Phase [deg]','X gain','Y gain','Z gain'],
      [(f'{phase:.0f}',*(f'{value:.4f}' for value in local))
       for phase,local in zip(fit['knot_phase_deg'],fit['knot_gains'])])
print('Global gains:',global_fit['gains'])

def save_fit(name, result):
    values={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    (OUT/name).write_text(json.dumps(values,indent=2)+'\\n')

save_fit('greybox_fit.json',fit)
save_fit('global_greybox_fit.json',global_fit)
'''),
('md', r'''
## Identification and held-out validation

Following Studios 5 and 11, minimize covariance-standardized **raw angular innovations** on training trajectories, updating only when an LSO is visible. No generated full-state observations or saved truth enter the fit. Filtered state histories are correlated outputs, so they support exploratory plots and guidance, not independent-measurement regression.

Both variants use the same training, validation and test trajectories. Lecture 13 motivates a local approximation when full-record stationarity is doubtful. Here windows are indexed by orbital phase, so training can pool repeated passages through the same phase. Adjacent gains are regularized towards one another, with no test-driven choice of window count. Independent test trajectories assess angular prediction and navigation accuracy. Free propagation uses its initial angular update and then the known commands, with no later measurements. Truth is used only to evaluate state errors.
'''),
('code', '''
scores={model:{split:sk.prediction_scores(orbit,rec,model_gains) for split,rec in data.items()}
        for model,model_gains in models.items()}
table(['Split','Model','Angular RMSE [arcsec]','Navigation position RMSE [km]','Navigation velocity RMSE [m/s]'],
      [(split,model,f"{q['angular_innovation_rmse_arcsec']:.3g}",
        f"{q['navigation']['position_rmse_km']:.3g}",f"{q['navigation']['velocity_rmse_m_s']:.3g}")
       for model,result in scores.items() for split,q in result.items() if split != 'training'])
(OUT/'prediction_scores.json').write_text(json.dumps(scores['Periodic gains'],indent=2)+'\\n')
(OUT/'model_comparison_scores.json').write_text(json.dumps(scores,indent=2)+'\\n')
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
global_pred=sk.predict(orbit,test,models['Global gains'])
global_residual=ld.residual_report(global_pred['standardized'],test.u,orbit.settings['residual_lags'])
table(['Model','Ljung–Box rejects','Eligible runs','Input–innovation p'],
      [(label,int(np.nansum(q['ljung_box_holm_p']<orbit.settings['alpha'])),
        q['eligible_whiteness_tests'],f"{q['ccf_global_p']:.3g}")
       for label,q in [('Global gains',global_residual),('Periodic gains',residual)]])
stationarity_rows=ld.stationarity_rows(pred['standardized'])
def fmt(p): return 'insufficient data' if p is None else f'{p:.3g}'
table(['Angular channel','Contiguous samples','ADF p','KPSS p','Joint reading'],[(r['channel'],r['contiguous_samples'],fmt(r['adf_p']),fmt(r['kpss_p']),r['reading']) for r in stationarity_rows])
(OUT/'innovation_stationarity.json').write_text(json.dumps(stationarity_rows,indent=2)+'\\n')
np.savez_compressed(OUT/'residual_checks.npz',**{k:v for k,v in residual.items() if isinstance(v,np.ndarray)})
(OUT/'optical_residual_summary.json').write_text(json.dumps({k:v for k,v in residual.items() if not isinstance(v,np.ndarray)},indent=2)+'\\n')
'''),
('md', r'''
## Do LSO-guided station keeping limits hold over a year?

Guidance uses only the LSO-derived departure estimate, with state updates at every measurement sample and planner calls at every measurement sample (about 1.5 hours). Candidate burns are checked by nonlinear propagation. The one-year run reports sampled corridor performance and the cumulative commanded Δv from the actual simulated impulses.

The comparison holds the initial departure, noise seed and measurement cadence fixed. The updated controller starts planning corrections earlier, using a target of 30% of the corridor limits instead of the baselines’ 70%. The two baselines use the reference-linearized navigation filter. The extended filter propagates the estimated state with the nonlinear dynamics, recomputes its local STM and visible angle Jacobians, and accounts for uncertainty in executed impulse magnitude. It uses the same phase-dependent actuator fit and no truth-state updates. These are paired outcomes from one simulated year, with completion reported separately from corridor compliance.
'''),
('code', '''
controllers={label:(g,'linear') for label,g in models.items()}
controllers['Periodic gains + extended navigation']=(gains,'extended')
trials={model:sk.stationkeeping_trial(
    orbit,model_gains, navigation=mode,
    duration_days=orbit.settings['annual_horizon_days'],
    planning_stride_steps=orbit.settings['annual_planning_stride_steps'],
    stop_on_departure_cutoff=False)
    for model,(model_gains,mode) in controllers.items()}
trial=trials['Periodic gains + extended navigation']
comparison={model:result['summary'] for model,result in trials.items()}
(OUT/'annual_model_comparison.json').write_text(json.dumps(comparison,indent=2)+'\\n')
table(['Model','Completed','First exit [day]','Annual commanded Δv [m/s]'],
      [(model,q['completed'],q['first_sampled_exit_day'],q['annual_delta_v_m_s'])
       for model,q in comparison.items()])
if (OUT/'uncontrolled.npz').exists():
    with np.load(OUT/'uncontrolled.npz') as z: no_day,no_y=z['day'],z['output']
else: no_day,no_y,_=sk.uncontrolled(orbit)
plots.controlled(orbit,trial,no_day,no_y,OUT/'figures')
plots.annual_delta_v(trial,OUT/'figures')
summary=trial['summary']
table(['Metric','Observed value'],[(k,summary[k]) for k in ('completed','requested_duration_days','last_day','planning_stride_steps','planning_margin','maximum_position_km','maximum_velocity_m_s','all_evaluated_samples_inside','first_sampled_exit_day','burns','total_commanded_delta_v_m_s','annual_delta_v_m_s','visible_epoch_fraction')])
np.savez_compressed(OUT/'closed_loop.npz',**{k:v for k,v in trial.items() if k!='summary'})
(OUT/'closed_loop_summary.json').write_text(json.dumps(summary,indent=2)+'\\n')
rejects=int(np.nansum(residual['ljung_box_holm_p']<.05))
display(Markdown(f'**Noise-model conclusion:** {rejects}/{residual["eligible_whiteness_tests"]} eligible whiteness tests reject after adjustment and the global residual–input p-value is {residual["ccf_global_p"]:.3g}, so these tests '+('flag predictor/noise inadequacy.' if rejects or residual['ccf_global_p']<.05 else 'do not reject the tested assumptions at 5%, without proving the model correct.')))
display(Markdown('**Station-keeping conclusion:** '+('both required norms remain inside their limits at all evaluated samples of the completed run.' if summary['completed'] and summary['all_evaluated_samples_inside'] else f'the extended-navigation controller does not establish the full corridor requirement, with observed peaks of {summary["maximum_position_km"]:.3g} km and {summary["maximum_velocity_m_s"]:.3g} m/s.')))
'''),
('md', r'''
## How often does the single-burn controller stay inside the corridor?

The annual Monte Carlo benchmark ran the periodic STM and extended navigation filter for 20 trials of 365.25 days. It varied the initial position and velocity errors and lunar surface observer angle noise. Initial errors had per-axis standard deviations of 30 m and 0.3 mm/s, and angle noise had a standard deviation of 1 arcsecond. The starting orbital phase, simulated dynamics and actuator gains were fixed.

| Error magnitude | Corridor limit | Peak of pointwise mean | Worst individual peak |
| --- | ---: | ---: | ---: |
| Position | 10 km | 2.545 km | 4.231 km |
| Velocity | 0.100 m/s | 0.0682 m/s | 0.1383 m/s |

The mean stayed inside both limits at every sampled epoch. **9 of 20 trials (45%)** stayed inside both limits for the full year. The other 11 trials crossed the velocity limit, with the earliest observed exit on day 35.4. The average uses the magnitude of each trial’s error at each epoch. Mean compliance and full-year trial compliance answer different questions.

Across all completed trials, annual commanded Δv averaged 6.87 m/s (range 6.48–7.38 m/s), with an average of 426 burns.

[Benchmark run and downloadable plot and trial data](https://github.com/scottblender/Halo_Orbit_Tracking/actions/runs/45171529580)
'''),
('md', r'''
## Next model discussion

Stationarity checks describe the records, while the phase-dependent impulse sensitivity motivates the periodic model structure. Held-out prediction, residual whiteness and annual corridor performance determine what the additional local gains are worth. Review LSO availability, gain uncertainty, raw angular residuals and navigation error before selecting a local OE/BJ model or periodic parameter basis (Studio 13). NARX remains deferred. Improving navigation or the burn planner may matter more than increasing black-box complexity.
''')]


def check_snapshot(path, expected):
    """Check the committed source and that it contains a successful execution."""
    saved = nb.read(path, as_version=4)
    nb.validate(saved)
    actual_source = [(cell.cell_type, cell.source) for cell in saved.cells]
    expected_source = [(cell.cell_type, cell.source) for cell in expected.cells]
    if actual_source != expected_source:
        raise SystemExit(f'{path.name}: sources changed. Rebuild, run and commit the notebook.')
    code = [cell for cell in saved.cells if cell.cell_type == 'code']
    if any(cell.execution_count is None or any(out.output_type == 'error' for out in cell.outputs)
           for cell in code):
        raise SystemExit(f'{path.name}: commit a successful execution of every code cell.')
    if not any('image/png' in out.get('data', {}) for cell in code for out in cell.outputs):
        raise SystemExit(f'{path.name}: commit the executed notebook with inline figures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check committed notebooks without overwriting them')
    args = parser.parse_args()
    for name, cells in [('01_stationkeeping_identification.ipynb', INTRO),
                        ('02_stm_greybox_stationkeeping.ipynb', MODEL)]:
        path = ROOT / 'notebooks' / name
        expected = notebook(cells)
        if args.check:
            check_snapshot(path, expected)
        else:
            nb.write(expected, path)
        print(name)
