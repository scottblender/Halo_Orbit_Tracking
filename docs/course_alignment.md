# Course methods and the LSO data path

The project asks whether LSO angular data can support prediction of impulse
magnitude and orbital phase while maintaining a 10 km position limit and a
0.1 m/s velocity limit. Position and inertial-velocity departures are resolved
on rotating Earth–Moon axes. The velocity conversion includes the transport
term from the rotating frame.

| Supplied studio | Use in this workflow |
| --- | --- |
| 03 Overfitting | Separate training, validation and test trajectories, with no adjacent-sample split |
| 04 LeastSquares | Scaled coordinates, least-squares impulse calculation and gain-sensitivity conditioning |
| 05 Likelihood | Gaussian angular-innovation criterion and conditional parameter uncertainty |
| 06 Stationarity | Correlation-aware segment statistics and paired ADF/KPSS tests |
| 07 ImpulseToSpectrum | Impulse/output alignment, lag analysis and excitation |
| 08 WelchInYourHands | Hann windows, 50% overlap, resolution sensitivity, MIMO spectral ratios and coherence |
| 09 ARMAPolesToModes | Physical interpretation of stability, using Floquet multipliers for this periodic orbit |
| 10 SameDataFourStructures | Separate dynamics accuracy from noise-model adequacy |
| 11 WatchingPEMIterate | Fit three physical actuator gains from raw angular prediction errors |
| 12 OrderSelectionBakeOff | Held-out residual checks and honest treatment of test support and multiple comparisons |
| 13 GlobalVsLocal | Orbital phase as the operating condition for a later local or periodic-basis model |

The angular measurement model, LSO site geometry, elevation and optical masks
reuse the existing repository functions. The diagnostic code adapts the supplied
autocovariance, segment-statistics, spectral and residual formulas. Student TODO
statistics are completed for this project. The uploaded class files are not
redistributed.

## Measurements and identification

Three lunar surface observatories generate noisy alpha/delta angles. Actual
geometry determines visibility, including horizon, range and configured
Sun/Earth exclusions. Missing angles remain NaN. The sampling cadence is recorded
explicitly as one uniform epoch per decision, approximately 1.5 hours.

The estimator starts from an independent nominal prior and uses the periodic
STM with a linearized angular observation map. It updates only for visible
stations and propagates its covariance through gaps. No direct position or
velocity observations are generated in the active notebooks.

Identification minimizes covariance-standardized raw angular innovations on
training trajectories. It does not regress on correlated state estimates or
read saved truth. The observation matrix is the angular Jacobian, not an
identity matrix. Validation and test use independent trajectory seeds. Truth
is used only to generate the simulated sensors and evaluate accuracy.

Welch plots use the reconstructed state record, which includes estimation
dynamics. Thus, these plots do not identify a plant-only global LTI transfer
function. Raw angular residual tests preserve actual time lags and independent
arc boundaries. Ljung–Box and ADF/KPSS use sufficiently long contiguous runs,
with insufficient support reported explicitly. Holm adjustment covers eligible
whiteness tests. A Gaussian Monte Carlo maximum-CCF test conditions on the actual
masks and fixed open-loop commands.

## Model conclusions and performance

The station-keeping model is nonlinear because the gravitational forces and
angular measurements depend nonlinearly on state. Its local departure dynamics
change with orbital phase, and the measurement channels change with geometry
and visibility. The underlying ideal CR3BP vector field is autonomous in
rotating coordinates. The STM along its periodic orbit is locally linear time
periodic. These statements describe different parts of the same system.

Uncontrolled propagation requests up to 24 orbits with no physical state resets.
A declared 20,000 km departure cutoff or an integration failure ends the
propagation early and is reported. Two orthographic projections avoid the
earlier overlapping 3-D labels. Every figure has a purpose sentence and a result
sentence computed from that run. Notebook 2 retains three figures and presents
the remaining checks in tables.

Guidance uses only the LSO-derived estimate to choose impulse magnitude and
phase. Nonlinear truth evaluates both sides of each impulse and coast substeps.
Observed corridor failures remain visible. Passing residual tests does not
prove controller success, and a single sampled trial does not establish
continuous-time satisfaction or a population success rate.

NARX remains deferred. The next model discussion should consider LSO coverage,
navigation accuracy, parameter uncertainty, phase sensitivity and residuals
before selecting a local OE/BJ model or periodic basis. Historical full-state
helpers remain only for isolated numerical tests. They are not the active
measurement path.
