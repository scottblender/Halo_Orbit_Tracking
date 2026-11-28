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
# 1. What do the LSO measurements tell us about halo-orbit station-keeping?

A near-rectilinear halo orbit (NRHO) is a repeating path near the Moon. Small position and velocity errors can grow from one orbit to the next. Three simulated **lunar surface observatories (LSOs)** measure the direction of the spacecraft. The estimator uses these angles and previous burn commands to infer the spacecraft's departure. The station-keeping planner then chooses when to burn and how much velocity to change to stay within **10 km** and **0.1 m/s** of the reference orbit.

| Signal | What it means |
| --- | --- |
| Input $u_k$ | Commanded three-axis velocity change $\Delta v$, in m/s |
| Measured output $z_{j,k}$ | Right ascension ($\alpha$) and declination ($\delta$) from visible LSO $j$ |
| Estimated state $e_k$ | Three position and three velocity departures from the planned orbit |

The identification data use independent, randomized open-loop burn commands to excite the three axes. The observation at epoch $k$ comes before burn $u_k$. Its first effect on a sampled angle can appear at $k+1$. The annual closed-loop guidance experiment uses a planner to select commands from the estimated departure.
'''),
('code', SETUP+'''
orbit,data=sk.prepare_study(ROOT,OUT/'data')
training=data['training']
table(['Split','Independent trajectories','Seed range'],[(n,len(r.seeds),f'{r.seeds[0]}–{r.seeds[-1]}') for n,r in data.items()])
print(f'LSO angle noise: {orbit.cfg["measurement"]["angle_sigma_arcsec"]:g} arcsec. Uniform cadence: {orbit.dt_s/3600:.2f} hour.')
table(['Site','Latitude [deg]','East longitude [deg]'],
      [(site['site']['name'],site['site']['latitude_deg'],site['site']['longitude_east_deg'])
       for site in orbit.sites])
plots.inputs_outputs(orbit,training,OUT/'figures')
'''),
('md', r'''
## What is actually measured?

The deterministic burn-response path **$G$** runs from commanded $\Delta v$ through the actuator gains, orbit dynamics and sightline geometry to the measured angles. At a given phase, the ideal angular response changes as the spacecraft moves and the LSO view changes.

$$z_{j,k}=h_j(t_k,x_k)+v_{j,k}$$

The noise path **$H$** includes additive sensor noise $v_{j,k}$. The simulator adds independent, zero-mean Gaussian noise with a standard deviation of 1 arcsecond to each ideal angle, then masks invisible observations. The filter also allows uncertainty in state propagation. A state disturbance can carry into future angles. Independent sensor draws describe the simulated sensor, while the filter's innovations reveal how well the full predictor accounts for expected errors.

We pool the training trajectories by adding their contributions to spectra or fits. Each trajectory keeps its own time axis, visibility gaps and starting state. Validation and test use different trajectories from training.
'''),
('code', '''
plots.lso_measurements(orbit,training,OUT/'figures')
plots.diagram(orbit,OUT/'figures')
'''),
('md', r'''
## Do the measured angles stay at a constant level?

Subtract the ideal angle on the reference orbit before testing. The resulting measured offset still includes real departure in the simulation and sensor noise. For each LSO angle we use its longest uninterrupted **training** record and show the seed, sample count and four-segment mean and variance checks. Gaps remain missing.

ADF tests a unit-root null and KPSS tests a level-stationarity null. At 5%, rejecting ADF while retaining KPSS supports a constant level. A KPSS rejection is evidence against a constant level. ADF and KPSS describe these records. The orbital model's phase dependence is evaluated separately using the response to burns.
'''),
('code', '''
angle_checks=ld.stationarity_rows(sk.measured_angle_offsets(orbit,training))
def p_value(value):
    return '—' if value is None else f'{value:.3g}'
table(['Measured offset','Training seed','Consecutive samples','ADF p','KPSS p*',
       'Mean flags / 4','Variance flags / 4','Joint reading'],
      [(r['channel'],training.seeds[r['arc']],r['contiguous_samples'],p_value(r['adf_p']),
        p_value(r['kpss_p']),r.get('mean_flags','—'),r.get('variance_flags','—'),r['reading'])
       for r in angle_checks])
print('* KPSS reports table bounds for extreme p-values. Segment flags use the approximate 95% bands.')
readings = {
    'supports level stationarity': 'The unit-root test rejects and the level-stationarity test retains its null. This run supports a constant level.',
    'conflicting tests': 'Both tests reject. A constant level is questionable even though ADF rejects a unit root.',
    'evidence against level stationarity': 'KPSS rejects a constant level and ADF cannot rule out a unit root. Treat a constant mean over this run as unsupported.',
    'inconclusive / low power': 'Neither test rejects. This run gives too little evidence to decide whether a constant level is reasonable.',
    'insufficient contiguous observations': 'Visibility leaves too few consecutive measurements for the tests.',
}
for r in angle_checks:
    name, angle=r['channel'].rsplit(' ',1)
    label='declination' if angle=='delta' else 'right ascension'
    reading=readings[r['reading']]
    if r.get('mean_flags',0) or r.get('variance_flags',0):
        reading+=f" The segment check flags {r['mean_flags']} means and {r['variance_flags']} variances out of four."
    display(Markdown(f'**{name} {label}.** {reading}'))
(OUT/'training_angle_stationarity.json').write_text(json.dumps(angle_checks,indent=2)+'\\n')
'''),
('md', r'''
## Welch spectra and empirical response

Following Studios 7–8, use Hann windows with 50% overlap. Pool windows within separate trajectories using the same sample cadence. The multi-input, multi-output frequency response (FRF) estimate is $\hat G_{\mathrm{nav}}=\hat S_{yu}\hat S_{uu}^{-1}$. Here $y$ is the LSO-derived state estimate. The FRF therefore includes filtering and coast intervals, and describes the command-to-navigation path. The physical command-to-angle path also depends on orbital phase and which LSOs are in view.

Check the spectral resolution, the number of averaged windows, conditioning of the three-input spectrum and coherence. Pairwise coherence indicates where a single command axis and the corresponding estimated position tend to vary together at the same frequency. The 0.8 threshold is an exploratory screen. Because the experiment spans a phase-varying orbit, use these plots to assess excitation and repeatable frequency content over the sampled records.
'''),
('code', '''
s=orbit.settings
windows = sorted({orbit.n//8,orbit.n//4,orbit.n//2,orbit.n,s['welch_length']})
spectra = {L:sd.welch_bundle(training.u,training.y[:,:-1],1/orbit.dt_s,L)
           for L in windows}
spec=spectra[s['welch_length']]
table(['Window samples','Resolution [cycles/day]','Averaged windows','Supported nonzero bins'],
      [(L,f"{86400/(orbit.dt_s*L):.3g}",q['segments'],
        f"{np.sum(q['supported'][1:])}/{len(q['supported'])-1}") for L,q in spectra.items()])
threshold=s['coherence_threshold']
table(['Direct command–position pair','Bins above coherence screen','Total supported nonzero bins'],
      [(axis,int(np.count_nonzero(spec['coherence'][1:,c,c][spec['supported'][1:]]>=threshold)),
        int(np.sum(spec['supported'][1:]))) for c,axis in enumerate('XYZ')])
cycle_per_day=86400/(orbit.n*orbit.dt_s)
display(Markdown(f"**Resolution and support.** The orbital fundamental is {cycle_per_day:.3f} cycles/day. "
                 f"The {spec['length']}-sample window resolves {86400/(orbit.dt_s*spec['length']):.3f} "
                 "cycles/day, so use it to examine broader frequency content. "
                 "A full-orbit window resolves the fundamental with one window per trajectory."))
plots.spectra(orbit,spec,OUT/'figures')
plots.frf(spec,OUT/'figures')
'''),
('md', r'''
## What changes with orbital phase?

We also check stationarity on the six estimated departure components of the first training trajectory. These values include filter updates when angles are available and propagated estimates during gaps.

To check the physics, apply the same small velocity perturbation at different phases and compare the resulting departures. Then compare a nonlinear propagation with the local state-transition matrix (STM) at several starting amplitudes. Finally, multiply the STMs over one orbit to obtain the **Floquet multipliers**, the periodic system's open-loop modes. A multiplier with magnitude above one signals growth each orbit. The time-varying and intermittently visible angle channels have no single global LTI pole-zero model.
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
## Do the baseline errors show a missing burn or noise response?

Use the physics-based STM with unit actuator gains on independent **validation** trajectories for a baseline. An innovation is the observed angle minus its one-step prediction. Scale the innovations jointly with the predicted covariance before applying two course checks.

- **$G$:** Compare past commands and innovations at their actual lags. Detectable correlation indicates that the predictor leaves some burn-related signal in the errors.
- **$H$:** Use Ljung–Box on the longest eligible contiguous runs and adjust the whiteness tests together with Holm's method. Detectable serial dependence means that the predictor leaves some time structure, which can reflect dynamics, measurement geometry or the noise model.

We keep visibility gaps and trajectory boundaries. The maximum-correlation screen redraws open-loop commands while holding the observed mask and innovations fixed. These diagnostics set a baseline for the fitted-model checks in Notebook 02.
'''),
('code', '''
validation=data['validation']
baseline=sk.predict(orbit,validation,np.ones(3))
baseline_report=ld.residual_report(baseline['standardized'],validation.u,s['residual_lags'])
baseline_p=baseline_report['ccf_global_p']
baseline_n=baseline_report['eligible_whiteness_tests']
baseline_rejects=int(np.count_nonzero(baseline_report['ljung_box_holm_p'] < s['alpha']))
table(['Path','Validation check','Unit-gain baseline'],
      [('G','Burn–innovation screen p',p_value(baseline_p)),
       ('H','Adjusted Ljung–Box rejections',f'{baseline_rejects}/{baseline_n}')])
if baseline_p is None:
    g_reading='Visibility leaves too few usable input–innovation pairs.'
elif baseline_p < s['alpha']:
    g_reading='Past commands still correlate with prediction errors. The baseline misses some burn-related response.'
else:
    g_reading='Past commands show no detectable association with the prediction errors at the tested lags.'
if not baseline_n:
    h_reading='No runs are long enough for the whiteness test.'
elif baseline_rejects:
    h_reading='Some eligible innovations retain serial dependence after adjustment.'
else:
    h_reading='The eligible innovation runs show no detectable serial dependence after adjustment.'
display(Markdown(f'**G.** {g_reading} **H.** {h_reading}'))
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

The nonlinear orbit model propagates the spacecraft, and the phase-dependent STM approximates small departures for estimation. Since the response to the same burn varies with phase, use that phase dependence in the physics of both gain models. The global model can pool all phases to fit one gain per axis. The local model can pool repeated passages near the same phase to estimate effective gains.

Notebook 02 compares the gain structures by training BIC and validation, checks $G$ and $H$ on independent test trajectories, and then asks whether either fitted model keeps the spacecraft inside the corridor with the same planner.
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
test=data['test']
predictions={label:(sk.predict(orbit,test,local),sk.predict(orbit,test,local,free_run=True))
             for label,local in models.items()}
pred,free=predictions['Periodic gains']
plots.compare_heldout(orbit,test,predictions,OUT/'figures')
np.savez_compressed(OUT/'test_predictions.npz',**pred,free_simulation=free['navigation'])
'''),
('md', r'''
## Do the residuals support the predictor?

Use standardized angular innovations with actual time lags and visibility gaps intact. Ljung–Box uses eligible contiguous runs with Holm adjustment. A mask-conditioned Monte Carlo maximum-CCF test covers the input/channel/lag family. No gaps or independent trajectories are concatenated, and no fitted ARMA degrees of freedom are subtracted for the fixed model on test data.

ADF and KPSS use the longest uninterrupted observed run for each channel. The table reports the channels that support testing. A coverage summary reports any channels left out. Residual whiteness checks the adequacy of the assumed noise model. Residual–input correlation checks whether commanded impulses remain related to the prediction error. Controller performance is evaluated separately using the corridor limits.
'''),
('code', '''
reports={label:ld.residual_report(updated['standardized'],test.u,orbit.settings['residual_lags'])
         for label,(updated,_) in predictions.items()}
residual=reports['Periodic gains']
plots.compare_angular_residuals(reports,OUT/'figures')
table(['Model','Ljung–Box rejects','Eligible runs','Input–innovation p'],
      [(label,int(np.nansum(q['ljung_box_holm_p']<orbit.settings['alpha'])),
        q['eligible_whiteness_tests'],f"{q['ccf_global_p']:.3g}")
       for label,q in reports.items()])
stationarity_rows=ld.stationarity_rows(pred['standardized'])
tested=[r for r in stationarity_rows if r['adf_p'] is not None and r['kpss_p'] is not None]
if tested:
    table(['Angular channel','Contiguous samples','ADF p','KPSS p','Joint reading'],
          [(r['channel'],r['contiguous_samples'],f"{r['adf_p']:.3g}",
            f"{r['kpss_p']:.3g}",r['reading']) for r in tested])
    if len(tested)==len(stationarity_rows):
        display(Markdown(f'**Angular stationarity coverage:** All {len(tested)} channels meet the contiguous-sample requirement. The table reports the periodic fit.'))
skipped=[r for r in stationarity_rows if r['adf_p'] is None or r['kpss_p'] is None]
if skipped:
    sites=', '.join(sorted({r['channel'].rsplit(' ',1)[0] for r in skipped}))
    counts=[r['contiguous_samples'] for r in skipped]
    coverage=str(min(counts)) if min(counts)==max(counts) else f'{min(counts)}–{max(counts)}'
    display(Markdown(f'**Angular stationarity coverage:** Tested {len(tested)} of {len(stationarity_rows)} channels. '
                     f'The {len(skipped)} skipped channels ({sites}) had longest observed runs of {coverage} '
                     f'consecutive samples. ADF and KPSS require at least {ld.ANGULAR_STATIONARITY_MIN_SAMPLES} here.'))
(OUT/'innovation_stationarity.json').write_text(json.dumps(stationarity_rows,indent=2)+'\\n')
np.savez_compressed(OUT/'residual_checks.npz',**{k:v for k,v in residual.items() if isinstance(v,np.ndarray)})
(OUT/'optical_residual_summary.json').write_text(json.dumps({k:v for k,v in residual.items() if not isinstance(v,np.ndarray)},indent=2)+'\\n')
'''),
('md', r'''
## Does the gain model change full-year station-keeping?

Guidance uses only the LSO-derived departure estimate, with state updates at every measurement sample and planner calls at every measurement sample (about 1.5 hours). Candidate burns are checked by nonlinear propagation. The one-year run reports sampled corridor performance and the cumulative commanded Δv from the actual simulated impulses.

Both runs use the revised extended navigation filter and the same single-burn planner, which targets 30% of the corridor limits. The filter propagates the estimated state with nonlinear dynamics, recomputes its local STM and visible angle Jacobians, and accounts for impulse uncertainty. The initial departure, angle-noise seed and measurement cadence are fixed. The side-by-side plots isolate the choice of a global or periodic actuator gain fit for this simulated year.
'''),
('code', '''
trials={model:sk.stationkeeping_trial(
    orbit,model_gains, navigation='extended', seed=26960,
    duration_days=orbit.settings['annual_horizon_days'],
    planning_stride_steps=orbit.settings['annual_planning_stride_steps'],
    stop_on_departure_cutoff=False)
    for model,model_gains in models.items()}
trial=trials['Periodic gains']
comparison={model:result['summary'] for model,result in trials.items()}
(OUT/'annual_model_comparison.json').write_text(json.dumps(comparison,indent=2)+'\\n')
def annual_metric(value):
    return '—' if value is None else f'{value:.3g}'
table(['Gains with revised filter','Completed','First exit [day]','Peak position [km]',
       'Peak velocity [m/s]','Burns','Commanded Δv [m/s]'],
      [(model,'Yes' if q['completed'] else 'No',
        'No exit' if q['first_sampled_exit_day'] is None else annual_metric(q['first_sampled_exit_day']),
        annual_metric(q['maximum_position_km']),annual_metric(q['maximum_velocity_m_s']),
        q['burns'],annual_metric(q['annual_delta_v_m_s']))
       for model,q in comparison.items()])
plots.compare_annual(orbit,trials,OUT/'figures')
summary=trial['summary']
table(['Metric','Observed value'],[(k,summary[k]) for k in ('completed','requested_duration_days','last_day','planning_stride_steps','planning_margin','maximum_position_km','maximum_velocity_m_s','all_evaluated_samples_inside','first_sampled_exit_day','burns','total_commanded_delta_v_m_s','annual_delta_v_m_s','visible_epoch_fraction')])
np.savez_compressed(OUT/'closed_loop.npz',**{k:v for k,v in trial.items() if k!='summary'})
(OUT/'closed_loop_summary.json').write_text(json.dumps(summary,indent=2)+'\\n')
rejects=int(np.nansum(residual['ljung_box_holm_p']<.05))
display(Markdown(f'**Noise-model conclusion:** {rejects}/{residual["eligible_whiteness_tests"]} eligible whiteness tests reject after adjustment and the global residual–input p-value is {residual["ccf_global_p"]:.3g}, so these tests '+('flag predictor/noise inadequacy.' if rejects or residual['ccf_global_p']<.05 else 'do not reject the tested assumptions at 5%, without proving the model correct.')))
compliant=[model for model,q in comparison.items() if q['completed'] and q['all_evaluated_samples_inside']]
status = ', '.join(compliant) if compliant else 'Neither gain model'
display(Markdown(f'**Full-year corridor compliance:** {status}. This is a paired comparison for one starting departure and one observation-noise seed.'))
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
