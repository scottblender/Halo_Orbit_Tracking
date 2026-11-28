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
from halo_tracking import lso_course_plots as course
ROOT=Path.cwd()
if not (ROOT/'configs').exists(): ROOT=ROOT.parent
OUT=ROOT/'artifacts'/'stationkeeping'
OUT.mkdir(parents=True,exist_ok=True)
plots.style()
def table(headers,rows):
    display(Markdown('| '+' | '.join(headers)+' |\\n| '+' | '.join(['---']*len(headers))+' |\\n'+
                     '\\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)))
'''



MEASURED_WELCH = '''
fs = 1/orbit.dt_s
length = orbit.settings['welch_length']
short = max(4, length//2)
training_offsets = sk.measured_angle_offsets(orbit, training)
angle_spectra = {L: course.angle_welch(training.u, training_offsets, fs, L)
                 for L in sorted({short, length})}
table(['Angle', *[f'{L} samples: windows / supported nonzero bins' for L in angle_spectra]],
      [(f'LSO {c//2+1} {("α", "δ")[c%2]}',
        *[f"{q[c]['segments']} / {np.count_nonzero(q[c]['supported'][1:])}"
           if q[c] is not None else '0 / 0' for q in angle_spectra.values()])
       for c in range(training_offsets.shape[-1])])
course.measured_frf(angle_spectra[short], short, fs,
                    orbit.settings['coherence_threshold'], OUT/'figures')
'''

MEASURED_WELCH_TEXT = r'''
### Welch and FRF directly from measured angles

Following Studios 7–8 and Lecture 8, estimate the MIMO response separately for each visible angle. Matched Hann windows stay within uninterrupted runs. Pair the pre-burn angle at epoch $k$ with command $u_k$ to retain the physical delay. Short windows give more visible windows at the expense of frequency resolution. These PSD, gain, phase and coherence estimates summarize the sampled phases of the orbit. Pairwise coherence is a screen for each input–output pair.
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

Subtract the ideal angle on the reference orbit before testing. The resulting measured offset still includes real departure in the simulation and sensor noise. For each LSO angle we use its longest uninterrupted **training** record and show the seed, sample count and four-segment mean and variance checks. Gaps remain missing. Following Studio 6 and Lecture 6, compare the across-trajectory mean $\mu[t]$, standard deviation $\sigma[t]$ and paired-angle covariance by orbital epoch. Plot each selected run's four segment means, variances, autocovariance and ACF. The inter-angle CCF compares the joint lag structure over those segments.

ADF tests a unit-root null and KPSS tests a level-stationarity null. At 5%, rejecting ADF while retaining KPSS supports a constant level. A KPSS rejection is evidence against a constant level. ADF and KPSS describe these records. The orbital model's phase dependence is evaluated separately using the response to burns.
'''),
('code', '''
offsets = sk.measured_angle_offsets(orbit, training)
angle_checks = ld.stationarity_rows(offsets)
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
course.ensemble_moments(orbit, offsets, 'training', OUT/'figures')
course.angle_segments(orbit, offsets, angle_checks, training.seeds, 'training', OUT/'figures')
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
('md', MEASURED_WELCH_TEXT),
('code', MEASURED_WELCH),
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
course.residual_checks({'Unit-gain baseline': baseline_report}, 'validation',
                       s['alpha'], OUT/'figures')
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

The spacecraft dynamics and the line-of-sight observation model are **nonlinear**. The STM and the varying sightline geometry give a **time-varying** local model around the reference orbit. Visible angle records show **nonstationary behavior**. The evidence varies by angle and interval. Since the response to the same burn varies with phase, use that phase dependence in the physics of both gain models. The global model can pool all phases to fit one gain per axis. The local model can pool repeated passages near the same phase to estimate effective gains.

Notebook 02 compares the gain structures by training BIC and validation, checks $G$ and $H$ on independent test trajectories, and then asks whether either fitted model keeps the spacecraft inside the corridor with the same planner.
'''),
('code', '''
response=np.linalg.norm(diagnostics['impulse_sensitivity'],axis=(1,2))
radius=float(abs(diagnostics['multipliers']).max())
amplitude = diagnostics['amplitudes_km']
error = np.median(diagnostics['linearity_relative_error'], axis=1)
nonstationary = ', '.join(r['channel'] for r in angle_checks
                          if r['kpss_p'] is not None and r['kpss_p'] < .05)
display(Markdown(f'**Nonlinearity.** The median dimensionless STM mismatch across phases '
                 f'changes from {error[0]:.2e} at {amplitude[0]:g} km to '
                 f'{error[-1]:.2e} at {amplitude[-1]:g} km perturbation scale.'))
display(Markdown(f'**Measured angles.** KPSS rejects level stationarity at 5% for '
                 f'{nonstationary or "no angles in the selected runs"}.'))
support=sum(r['adf_p']<.05 and r['kpss_p']>=.05 for r in stationarity_rows)
display(Markdown(f'**Observed evidence:** horizon sensitivity varies by a factor of {response.max()/response.min():.2g} across phase, the largest one-orbit multiplier is {radius:.3g}, and {support}/6 reconstructed state channels give the ADF/KPSS combination supporting level stationarity.'))
display(Markdown('**Stability conclusion:** '+('the multiplier outside the unit circle identifies a locally unstable mode, and the extended uncontrolled trajectory shows its finite-time consequences.' if radius>1.001 else 'these multiplier calculations do not robustly resolve instability.')))
''')]


MODEL = [
('md', r'''
# 2. Can lunar angle measurements support halo-orbit station-keeping?

Three simulated **lunar surface observatories (LSOs)** track a spacecraft in a near-rectilinear halo orbit (NRHO). Each visible LSO reports two line-of-sight angles: right ascension ($\alpha$) and declination ($\delta$). The estimator combines these observations with past burn commands to infer the spacecraft's **position and velocity departure** from the reference orbit. The planner uses that estimate to choose the timing and three-axis velocity change ($\Delta v$) of a corrective burn. The corridor allows up to **10 km** of position departure and **0.1 m/s** of velocity departure.

### Where the command and noise enter

- **Burn path ($G$).** The commanded burn $u_k$ changes velocity through actuator gains $g(\phi_k)$. Gravity propagates the resulting departure $e$ and the LSO geometry maps it to angles. The LSO measurement at epoch $k$ comes before the burn at $k$, so the first affected sampled angle is at $k+1$.
- **Noise path ($H$).** The simulator adds independent Gaussian sensor noise ($1$ arcsec standard deviation) to each ideal angle and then applies the visibility mask. The filter also allows state prediction uncertainty through $Q$, including burn uncertainty in the annual extended filter. A state disturbance can propagate into future angles, even when the sensor noise itself is white.

$$
e_{k+1}=A_k e_k+B_k\operatorname{diag}(g(\phi_k))u_k+w_k,\qquad
z_{j,k}-\bar z_{j,k}\approx C_{j,k}e_k+v_{j,k}.
$$

$A_k$ is the state-transition matrix (STM) that propagates a small departure for one sample. $B_k$ maps the burn through that step. $C_{j,k}$ is the local slope of the sightline angles with respect to the departure. The filter uses only its visible rows. The nonlinear simulation supplies the spacecraft truth for scoring and generating synthetic observations. The fit uses the measured angles and commands. A grey box is appropriate because the orbit and sensor geometry give $A_k$, $B_k$ and $C_{j,k}$ directly, leaving the actuator gains to be estimated from data.
'''),
('md', r'''
## 1. Are the measured angles stationary?

First subtract the angle predicted on the reference orbit. This accounts for the nominal sightline motion and leaves the **measured angle offset**, which still contains spacecraft departure and sensor noise. We use the longest uninterrupted test run for each angle and print its trajectory seed and length. We also compare means and variances across four subsegments with correlation-aware bands, as in Studio 6. A flag counts a subsegment outside the approximate 95% band around the segment average. The ensemble plots show $\mu[t]$, $\sigma[t]$ and paired-angle covariance at each epoch across test trajectories. The segment autocovariance, ACF and inter-angle CCF show the lag structure across four parts of a contiguous run.

At the 5% level, **ADF** tests a unit-root null and **KPSS** tests a constant-level stationarity null. ADF rejection plus KPSS non-rejection supports level stationarity for that run. If both reject, some behavior departs from the simple level-stationary model even though ADF rejects a unit root. The tests describe these observed records. The phase dependence of the spacecraft dynamics is a separate question.
'''),
('code', SETUP+'''
orbit,data=sk.restore_study(ROOT,OUT/'data')
test = data['test']
alpha = orbit.settings['alpha']
offsets = sk.measured_angle_offsets(orbit, test)
angle_rows = ld.stationarity_rows(offsets)

def test_p(value):
    return '—' if value is None else f'{value:.3g}'

table(['Measured offset','Test seed','Consecutive samples','ADF p','KPSS p*',
       'Mean flags / 4','Variance flags / 4','Reading'],
      [(r['channel'],test.seeds[r['arc']],r['contiguous_samples'],
        test_p(r['adf_p']),test_p(r['kpss_p']),r.get('mean_flags','—'),
        r.get('variance_flags','—'),r['reading']) for r in angle_rows])
print('* KPSS reports table bounds when the p-value falls outside its tabulation.')
readings = {
    'supports level stationarity': 'The run supports a constant level by ADF and KPSS. Check the segment flags for changes these two tests may miss.',
    'conflicting tests': 'ADF rejects a unit root and KPSS rejects level stationarity. A model with a constant mean over this run is questionable.',
    'evidence against level stationarity': 'KPSS rejects a constant level and ADF cannot rule out a unit root. Treat a constant level over this run as unsupported.',
    'inconclusive / low power': 'Neither test rejects. The run gives too little evidence to decide whether a constant level is reasonable.',
    'insufficient contiguous observations': 'Visibility leaves too few consecutive angles for these tests.'
}
for r in angle_rows:
    name, angle = r['channel'].rsplit(' ',1)
    flags = ''
    if r.get('mean_flags', 0) or r.get('variance_flags', 0):
        flags = (f" The segment check flags {r['mean_flags']} means and "
                 f"{r['variance_flags']} variances out of four.")
    display(Markdown(f"**{name} {angle} ({'declination' if angle == 'delta' else 'right ascension'}).** "
                     + readings[r['reading']] + flags))
course.ensemble_moments(orbit, offsets, 'test', OUT/'figures')
course.angle_segments(orbit, offsets, angle_rows, test.seeds, 'test', OUT/'figures')
(OUT/'measured_angle_stationarity.json').write_text(json.dumps(angle_rows,indent=2)+'\\n')
'''),
('md', r'''
## 2. How does a burn change the measured angles?

$G$ is a **phase-dependent** command-to-angle map. For a burn at epoch $k$, the first-sample map is $C_{k+1}B_k\operatorname{diag}(g_k)$. At later samples, the intervening STMs propagate that same departure before the observation map acts. The table compares the ideal responses at several equally spaced burn phases. An LSO supplies those angles to the filter only when visible.

The eigenvalues of the product of STMs over one complete orbit are **Floquet multipliers**. They describe the modes that grow, decay, or oscillate from orbit to orbit. Both gain fits share these open-loop modes because the fitted gains appear in $B_k\operatorname{diag}(g_k)$, while the homogeneous dynamics depend on $A_k$. A single set of LTI transfer-function poles and zeros would require time-invariant channels. A lifted periodic input–output zero analysis would also require specifying a repeating measurement schedule and the input/output channels, so this study reports the physical modes and the phase-dependent response directly.
'''),
('code', '''
from halo_tracking.measurements import ARCSEC

horizon = orbit.settings['horizon_steps']
burn_phases = np.arange(4) * orbit.n // 4
response_rows = []
for k in burn_phases:
    response = orbit.B[k] * 1e-3  # unit test input of 1 mm/s on each axis
    _, c_next = orbit.observation_terms(k + 1)
    first = np.linalg.norm(c_next @ response) / ARCSEC
    for j in range(1, horizon):
        response = orbit.A[(k+j) % orbit.n] @ response
    _, c_end = orbit.observation_terms(k + horizon)
    last = np.linalg.norm(c_end @ response) / ARCSEC
    response_rows.append((360*k/orbit.n, first, last))
table(['Burn phase [deg]','First sample [arcsec]',f'Sample +{horizon} [arcsec]'],
      [(f'{p:.0f}',f'{a:.3f}',f'{b:.3f}') for p,a,b in response_rows])
display(Markdown('Each value combines the ideal six-angle responses to three separate 1 mm/s '
                 'axis burns using unit actuator gains. The change across rows shows that '
                 'the same commanded burn can yield different angles at different orbit phases.'))
monodromy = np.eye(6)
for transition in orbit.A:
    monodromy = transition @ monodromy
modes = sorted(np.linalg.eigvals(monodromy),key=abs,reverse=True)
table(['Mode','Multiplier','Magnitude','Phase [deg/orbit]'],
      [(i,f'{m:.5g}',f'{abs(m):.5g}',f'{np.angle(m,deg=True):.3g}')
       for i,m in enumerate(modes,1)])
display(Markdown(f'**Open-loop stability.** The dominant mode grows by about {abs(modes[0]):.3g}× '
                 'per orbit in the local model. This growth motivates corrective burns.'))
(OUT/'phase_response.json').write_text(json.dumps(dict(
    response=response_rows, multipliers=[[m.real,m.imag] for m in modes]),indent=2)+'\\n')
'''),
('md', r'''
## Welch and FRF checks on the training commands

The Welch estimate below uses the known training commands and the reconstructed position from the baseline LSO filter. It checks how much frequency resolution and command-to-estimate coherence the record provides. The estimated FRF includes the filter and samples from different orbital phases. Section 2 computes the physical phase-dependent command-to-angle map. The held-out residual section checks $G$ and $H$ using measured angles.
'''),
('code', '''
training = data['training']
welch = {}
for L in sorted({orbit.settings['welch_length'], orbit.n}):
    welch[L] = sd.welch_bundle(training.u, training.y[:,:-1], 1/orbit.dt_s, L)
table(['Window samples','Resolution [cycles/day]','Windows','Supported nonzero bins',
       'X/Y/Z bins meeting coherence screen'],
      [(L, f'{86400/(orbit.dt_s*L):.3g}', q['segments'],
        f"{np.sum(q['supported'][1:])}/{len(q['supported'])-1}",
        ' / '.join(str(np.count_nonzero(q['coherence'][1:,c,c][q['supported'][1:]]
                       >= orbit.settings['coherence_threshold'])) for c in range(3)))
       for L,q in welch.items()])
print(f"The pairwise coherence screen is {orbit.settings['coherence_threshold']:.2f}. "
      "Short windows yield more averages. Full-orbit windows resolve one cycle per orbit.")
spec = welch[orbit.settings['welch_length']]
plots.spectra(orbit, spec, OUT/'figures')
plots.frf(spec, OUT/'figures')
'''),
('md', MEASURED_WELCH_TEXT),
('code', MEASURED_WELCH),
('md', r'''
## 3. Do the trajectories support phase-dependent actuator gains?

The **global-gain** model pools all training trajectories and phases into one gain per burn axis. The **periodic-gain** model pools independent passes through the same orbital phases, interpolates between six overlapping phase windows, and smooths neighboring gains across the end of the orbit. Both use phase-dependent STMs and measurement geometry. The simulator uses constant actuator gains, so phase-dependent fitted gains can absorb discrepancies between the nonlinear simulation and the linearized predictor.

The fit minimizes joint, covariance-standardized one-step angular innovations on eight open-loop training trajectories. Four independent trajectories support validation and four support the final test. For the BIC comparison, fit the phase-window alternatives **without** smoothing on the same training epochs. The nominal BIC difference combines the Gaussian innovation sum of squares and the parameter penalty $d\log N$ (with $N$ measured epochs and $d$ gain coefficients). The innovation covariance schedule is fixed across the candidates, so its log-determinant cancels. The criterion is a relative comparison under the working innovation model. Short records and residual dependence limit its interpretation. The smoothed six-window candidate is evaluated separately with the global model for the test and annual trial.
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

training = data['training']
window_counts = sorted({3, fit['windows']})
n_epochs = int(training.visible.any(axis=-1).sum())
candidates = [('Global',3,global_fit['cost'],global_fit['gains'])]
for windows in window_counts:
    candidate = sk.fit_periodic_greybox(orbit, training, windows=windows, smoothing=0)
    candidates.append((f'{windows} phase windows',3*windows,
                       candidate['training_cost'],candidate['gains']))
selection = []
for name, d, cost, candidate_gains in candidates:
    validation = sk.predict(orbit, data['validation'], candidate_gains)['standardized']
    selection.append(dict(model=name, d=d, training_sum_squares=2*cost,
                          bic_relative=2*cost+d*np.log(n_epochs),
                          validation_rms=float(np.sqrt(np.nanmean(validation**2)))))
best = min(selection,key=lambda r:r['bic_relative'])
table(['Training candidate','Gain coefficients','Training sum of squares','ΔBIC','Validation RMS [standardized]'],
      [(r['model'],r['d'],f"{r['training_sum_squares']:.1f}",
        f"{r['bic_relative']-best['bic_relative']:.2f}",f"{r['validation_rms']:.3f}") for r in selection])
best_validation = min(selection,key=lambda r:r['validation_rms'])
display(Markdown(f"**Model selection.** The lowest nominal BIC is **{best['model']}**. "
                 f"The lowest validation innovation RMS is **{best_validation['model']}**. "
                 "Lower BIC favors explaining the data with fewer gain coefficients. "
                 "The validation score checks how well those gains predict angles on new trajectories."))
(OUT/'gain_model_selection.json').write_text(json.dumps(selection,indent=2)+'\\n')
'''),
('md', r'''
## 4. Do the angular updates improve held-out navigation?

Compare each model's state estimate against the simulator's truth. The free prediction uses the first angular update and the known burn commands, then coasts without more angles. The updated estimate uses angles every time an LSO is visible. The angular RMSE below is the one-step error **before** using that epoch's measurement.
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
for label, q in scores.items():
    pos = q['test']['navigation']['position_rmse_km']
    coast = q['test']['free_run']['position_rmse_km']
    display(Markdown(f"**{label}.** Test position RMSE is {pos:.3f} km with angle updates "
                     f"and {coast:.3f} km in free prediction."))
'''),
('md', r'''
## 5. Do the residuals support the burn path $G$ and the noise path $H$?

An **innovation** is a measured angle minus its one-step prediction. For this check, the filter standardizes the innovations jointly using its predicted covariance, including correlations among visible angles at the same epoch. These standardized coordinates can mix visible angles, so the whiteness check refers to predictor coordinates. Section 1 reports the individual measured angles.

- **$G$, input–output dynamics.** If past commands still correlate with the innovation, the model may have missed a burn response. We correlate $u_k$ with the innovation at $k+1$ and later, across all three axes and all supported standardized coordinates. We redraw the independent open-loop commands for a maximum-correlation reference while keeping residuals and masks fixed. This screen preserves residual serial dependence and uses observed visibility as fixed. Its conclusion applies to the tested excitation, lags and measurements. The simulated visibility can depend on the trajectory, so this is a screening p-value conditional on the observed mask.
- **$H$, noise and uncertainty.** If the filter models the predictable part of the angle errors, the innovations should have little residual serial correlation. Ljung–Box tests the longest uninterrupted run in each eligible test trajectory and coordinate. Holm adjustment accounts for the eligible family within each model. The error plots use original sample lags and omit missing pairs. A rejection can reflect dynamics, uncertainty tuning, or the measurement model, even though the simulated sensor noise is white by construction.
'''),
('code', '''
reports={label:ld.residual_report(updated['standardized'],test.u,orbit.settings['residual_lags'])
         for label,(updated,_) in predictions.items()}
residual=reports['Periodic gains']
course.residual_checks(reports, 'heldout', alpha, OUT/'figures')
diagnostic_rows = []
for label, q in reports.items():
    adjusted = q['ljung_box_holm_p']
    rejected = np.argwhere(adjusted < alpha)
    n = q['eligible_whiteness_tests']
    p = q['ccf_global_p']
    diagnostic_rows.append(dict(model=label,rejects=len(rejected),eligible=n,
                                g_p=p,max_command_lag_samples=q['lags']+1))
    if n == 0:
        h_reading = 'Visibility left no runs long enough to test whiteness.'
    elif len(rejected):
        runs = ', '.join(f'seed {test.seeds[a]}, coordinate {c+1} (adjusted p={adjusted[a,c]:.3g})'
                         for a,c in rejected)
        h_reading = f'Some prediction errors retain time structure. Rejected runs: {runs}.'
    else:
        h_reading = 'The eligible runs show no detectable serial dependence at the tested lags.'
    if p is None:
        g_reading = 'Too few usable command–innovation pairs to test the burn response.'
    elif p < alpha:
        g_reading = 'Past burns still predict part of the innovation. Revisit the command-to-angle model.'
    else:
        g_reading = 'No detectable command–innovation correlation over the tested lags.'
    display(Markdown(f'**{label}.** **G:** {g_reading} **H:** {h_reading}'))
table(['Model','G: command–innovation p','H: Ljung–Box rejections','Eligible runs'],
      [(r['model'],test_p(r['g_p']),r['rejects'],r['eligible']) for r in diagnostic_rows])
print('Eligible runs use at least 24 consecutive samples. The input screen covers command-to-angle '
      f'delays of 1–{diagnostic_rows[0]["max_command_lag_samples"]} samples.')
(OUT/'gain_model_diagnostics.json').write_text(json.dumps(diagnostic_rows,indent=2)+'\\n')
# Retain an angle-by-angle stationarity check for the periodic predictor.
innovation_rows = ld.stationarity_rows(pred['residual'] / ARCSEC)
display(Markdown('**Additional check.** ADF and KPSS on the raw periodic-model '
                 'innovations describe their level behavior. Ljung–Box above tests whiteness.'))
table(['Raw innovation','Test seed','Consecutive samples','ADF p','KPSS p*','Reading'],
      [(r['channel'],test.seeds[r['arc']],r['contiguous_samples'],test_p(r['adf_p']),
        test_p(r['kpss_p']),r['reading']) for r in innovation_rows])
(OUT/'innovation_stationarity.json').write_text(json.dumps(innovation_rows,indent=2)+'\\n')
np.savez_compressed(OUT/'residual_checks.npz',**{k:v for k,v in residual.items() if isinstance(v,np.ndarray)})
(OUT/'optical_residual_summary.json').write_text(json.dumps({k:v for k,v in residual.items() if not isinstance(v,np.ndarray)},indent=2)+'\\n')
'''),
('md', r'''
## 6. Do the burn commands keep the spacecraft within the corridor?

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
for model, q in comparison.items():
    exit_day = q['first_sampled_exit_day']
    if exit_day is not None:
        reading = (f'First corridor violation on day {exit_day:.2f}. '
                   'Cumulative commanded Δv includes burns after the violation.')
    elif q['completed']:
        reading = 'Every evaluated sample stayed inside both corridor limits for the full year.'
    else:
        reading = 'The run ended before the full year without an observed corridor violation.'
    display(Markdown(f'**{model}.** {reading}'))
display(Markdown('The paired comparison uses one starting departure and one observation-noise seed.'))
'''),
('md', r'''
## 7. What the grey box can support

The spacecraft dynamics and sightline observations are **nonlinear**. The local model is **time-varying** even with constant actuator gains. Section 2 shows how the same burn produces different angular responses at different phases. The visible angle records show **nonstationary behavior**, with channel-specific evidence in the moment, segment and ADF/KPSS results in Section 1.

The grey box supports orbit estimation in this experiment. The angular updates improve held-out position and velocity estimates, and the global gain fit is the simpler baseline favored by BIC and validation. The held-out tests find no detectable dependence on past burns or violations of whiteness in the eligible standardized innovation runs. These findings apply to the measured phases and lags. The paired annual trial has one starting departure and one noise seed, so it gives an example of corridor performance for each model.

For guidance, compare the single-burn planner against **output-feedback model predictive control (MPC)**, which can optimize several future burns under corridor and command constraints and replan as new estimates arrive. Also benchmark the differential-correction and minimization variants of x-axis crossing targeting, with and without covariance-aware prediction, from Shimane et al. (arXiv:2512.01182v4). Use matched starting departures and noise seeds to compare full-year corridor passes and cumulative commanded $\Delta v$.

For identification, investigate phase- and visibility-dependent innovation structure before adding parameters. A local output-error (OE) benchmark assumes noise at the angle output. A local Box–Jenkins (BJ) benchmark can model a separate colored disturbance path if the records support estimating it. Compare both paths on independent trajectories in the same phase and visibility ranges.
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
