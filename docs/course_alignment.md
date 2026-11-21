# Course alignment and experiment scope

The active project asks whether an identified, phase-dependent departure model
can plan impulse magnitude and orbital phase while maintaining **3-D** position
error below **10 km** and inertial velocity error below **0.1 m/s**. Both errors
compare the actual spacecraft with the nominal spacecraft at the same epoch;
phase is time through the corrected orbital period, not a geometric angle.

## Methods from the supplied class notebooks

| Supplied studio | Applied here | Adaptation |
| --- | --- | --- |
| 03 Overfitting | Separate estimation, validation and test trajectories | No random split of adjacent samples; no selection on test results |
| 04 LeastSquares | `lstsq`, sensitivity singular values, scaled coordinates | Three physical actuator gains rather than a high-order polynomial |
| 05 Likelihood | Gaussian innovation likelihood | Declared, time-dependent covariance; conditional standard errors |
| 06 Stationarity | `acov`, `segment_stats`, paired ADF/KPSS tests | Six outputs and three inputs; separate experiments; finite-record caveats |
| 07 ImpulseToSpectrum | Impulses, lags, spectra and excitation | Three commands, six departures; pre-burn output alignment |
| 08 WelchInYourHands | Hann/50% Welch, resolution sensitivity, coherence, conservative PSD DOF | Full MIMO spectral inversion; one-sided SciPy density convention; no global LTI interpretation |
| 09 ARMAPolesToModes | Stability and physical interpretation of poles | One-orbit Floquet multipliers for periodic dynamics; no forced stable pole reflection |
| 10 SameDataFourStructures | Separate dynamics and noise validation | Empirical/model response comparisons separate from innovation whiteness |
| 11 WatchingPEMIterate | Prediction-error estimation | Periodic physical state-space predictor and three unknown gains |
| 12 OrderSelectionBakeOff | Held-out checks, Ljung–Box, residual-input CCF | Complete the student TODO statistics; Holm families; no unjustified ARMA df subtraction on held-out data |
| 13 GlobalVsLocal | Phase as operating condition | Phase-sensitive STM is the starting model; later local/periodic-basis fits remain open |

`stationkeeping_diagnostics.py` adapts the supplied Studio 06 autocovariance,
Bartlett-taper segment statistics and ADF/KPSS logic. The windowed Fourier
products and conservative non-overlapped DOF follow Studio 08. Residual ACF
and Ljung–Box/CCF formulas follow Studios 10/12. Student TODOs were completed
for this project; no unfilled class exercise is invoked. The orbit integrator,
analytic Jacobian and nominal differential correction reuse this repository's
existing `dynamics.py` and `orbit_reference.py`. The uploaded course notebooks
are not redistributed in this public repository.

## Data and prediction protocol

- Correct the rounded catalog seed before constructing a periodic reference.
- Generate fresh independent **nonlinear** open-loop trajectories; old tracking
  and NARX teacher artifacts use different tolerances and are not reused.
- Use uniform pre-burn observations. An impulse `u[k]` precedes propagation to
  `y[k+1]`; no terminal observation is relabeled as an earlier response.
- Position uses rotating axes. Velocity is the inertial departure resolved
  along rotating axes, including the angular transport term. This keeps the
  velocity corridor a physical inertial-velocity requirement. The matrix `S`
  explicitly performs this conversion.
- The full-state observation is synthetic navigation data. It is not a claim
  about lunar-surface optical observability or Gateway tracking.
- Inputs are independently generated commanded impulses. Unknown diagonal
  actuator gains deliberately introduce an identifiable physical mismatch.
  The estimator uses **only commands and noisy outputs**, never saved truth
  or the hidden simulated gains. Those gains are used only by the plant.
- Estimate three gains on training data with fixed dynamics, fixed navigation
  covariance, and a small fixed model-error floor. Validate unity versus fitted
  gain behavior and report test prediction/free-simulation errors separately.
- Do not join independent trajectories in Welch, ACF or CCF calculations.
  Checksum data and store all seeds, frame conventions, configuration and units.

## Interpreting diagnostics

The nonlinear autonomous plant, local periodically varying departure dynamics,
and nonlinear closed-loop guidance are different systems. A stationary random
input does not make an unstable/periodic response stationary. ADF/KPSS only
screen a particular short record; they do not establish wide-sense stationarity.
KPSS warnings retain whether the p-value is bounded by the lookup table.

Full-orbit Welch/FRF curves are exploratory summaries. Pairwise coherence can
be low because several independent inputs drive one output, even with a good
MIMO model. Full spectral inversion and conditioning checks distinguish this
from weak input support. The input PSD band uses the Studio 08 conservative
non-overlapped count; output/FRF bands are not asserted for nonstationary data.
The 0.8 coherence threshold is a declared screen, not a significance test.
The grey-box FRF comparison uses a free simulation driven by the same commands,
not a fictitious global LTI transfer function. It complements time-domain
validation and does not by itself select the best model class.

Residual whiteness is tested on covariance-standardized, held-out one-step
innovations. Ljung–Box is per trajectory/channel, with the trained model fixed
(no fitted ARMA degree subtraction). Holm adjustment covers that whole family;
a second Holm family covers all causal input/innovation lag/channel pairs.
Pointwise plot bands do not carry the Holm interpretation. These are nominal
large-sample screens; failure may reflect linearization error or an inadequate
noise model, and success is not proof of correct dynamics.

## Guidance and performance limits

The fitted plant drives a receding-horizon, single-impulse planner. It considers
no burn and several candidate waits, solves a weighted path correction and
checks predictions with nonlinear propagation. It selects timing itself, without
reference burn labels. Initial conditions, trial seed, horizon and candidate
grid are declared before evaluation. A pending delayed burn prevents duplicate
scheduling. Performance uses nonlinear **truth**, distinct from the noisy state
estimate used for guidance.

Both pre/post impulse states and coast substeps are checked against norm limits;
an impulse cannot repair an existing position violation instantaneously. The
summary reports observed maxima and the first violation instead of declaring
success from RMSE. Grid sampling and one initial-condition/noise trial cannot
establish continuous-time satisfaction or a population success rate. One
impulse over the chosen horizon may be infeasible; failures are retained.
The same-physics synthetic mismatch study also cannot establish robustness to
solar perturbations, ephemeris errors, navigation outages or actuator limits
beyond those explicitly modeled. Those are later experiments, not hidden claims.

## Next model discussion

Review phase-conditioned sensitivity, input conditioning, finite-amplitude
linearity errors, held-out residuals and empirical/model response agreement.
Then consider local phase-conditioned OE/BJ models, or a periodic basis model.
No NARX architecture or black-box order is selected in this overhaul.
