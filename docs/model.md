# Model and uncertainty definitions

## Normalized state

The propagated state is x = [r, v] in Earth–Moon barycentric rotating coordinates, with Earth at (−μ, 0, 0) and Moon at (1−μ, 0, 0). Rotating angular speed is one radian per TU. Dimensional scales are LU, LU/TU, and LU/TU²; the supplied period is in TU. The stability index is metadata, not a covariance or estimator tuning parameter.

The dynamics use the standard CR3BP gravitational, centrifugal and Coriolis terms. The analytic Jacobian drives dΦ/dt = AΦ. A period-closure diagnostic does not correct the input state. Jacobi conservation checks numerical integration, not whether the orbit is periodic or belongs to the labeled family.

## Frames and acceleration

Let q = r − [1−μ,0,0] and Wq = [0,0,1] × q. With R(t) a positive rotation about z,

- r_MCI = LU R q
- v_MCI = (LU/TU) R (v + Wq)
- a_MCI = (LU/TU²) R (a_rot + 2Wv + W²q)

MCI here means model Moon-centered, nonrotating axes, with orientation defined at t=0. It is an accelerating-origin frame, so acceleration is relative to the Moon. Its axes are not tied to an astronomical epoch. Covariances transform with the full state Jacobian including velocity dependence on position. Acceleration covariance is Ga P Gaᵀ, where Ga differentiates the physical relative acceleration with respect to the normalized rotating state. In this force model the physical acceleration has no velocity dependence, providing an independent transformation check.

The body-fixed site is constant in synodic coordinates under ideal synchronous rotation. It still has nonzero inertial velocity. Site position is assumed exactly known. Minimum elevation ≥0 supplies a spherical lunar horizon constraint; Earth occultation and optical exclusions are not implemented.

## Measurements

Using inertial relative position ρ = r_sc − r_site and velocity u = v_sc − v_site, set q_h = sqrt(ρx² + ρy²):

- α = atan2(ρy, ρx)
- δ = atan2(ρz, q_h)
- αdot = (ρx uy − ρy ux) / q_h²
- δdot = [q_h uz − ρz(ρx ux + ρy uy)/q_h] / |ρ|²

Angles use radians; rates use radians per physical second. Alpha residuals wrap across ±π. RA-like coordinate singularities raise an error rather than silently producing invalid rates. Measurement Jacobians use centered differences in normalized coordinates and are checked against independent directional perturbations. This is suitable for the baseline, with a tangent-plane direction representation a possible later improvement near poles.

AO and AO+rates use the same angle observations and initial error. With correlated same-epoch channels, supply the full 4×4 covariance in rad/rad-per-second units; AO uses its leading 2×2 marginal. This does not support temporally correlated differenced rates.

## Estimator

The EKF propagates six states and covariance, uses the actual visibility schedule, wraps angular residuals and applies Joseph covariance updates. Gaps trigger propagation without measurement updates. Initial P0 is specified in normalized rotating coordinates, with position and velocity sigmas converted from km and km/s. There is no hidden use of the true initial state beyond defining the simulated nominal prior mean; the filter receives a seeded random prior error.

The baseline uses per-axis 1σ uncertainties of 1 km and 0.01 m/s, with zero cross-covariances. In dimensional rotating-state units of km and km/s, P0 = diag(1, 1, 1, 1e-10, 1e-10, 1e-10); the implementation converts this to normalized LU/TU coordinates. Each angle has 1σ = 1 arcsec = 4.848136811e-6 rad, so the angles-only R = 2.350443054e-11 I₂ rad². Independent angular-rate noise remains 0.01 arcsec/s per component. One nominal period is 1.8572 TU, approximately 8.06485 days with the configured TU.

Optional acceleration PSD has units LU²/TU³, in normalized rotating velocity derivatives. A local covariance integral W follows dW/dt = AW + WAᵀ + L. This baseline provides exact model-matched deterministic truth when the PSD is zero.

Covariance-only mode linearizes along nominal truth and performs zero-innovation updates. It predicts local covariance; its zero state error and zero NIS are not measured estimator performance.

## Validation and interpretation

- Plot actual errors together with ±3σ component bounds.
- RSS σ = sqrt(trace(P_position)) or sqrt(trace(P_velocity)); this is not a one-dimensional directional bound.
- NIS uses 2 or 4 measurement degrees of freedom; NEES uses six state degrees of freedom. Expected means are approximate linear-Gaussian reference values, not guarantees for this nonlinear scenario.
- Monte Carlo varies prior errors and measurement noise, preserving pairing between measurement cases. It returns epochwise NEES, NIS samples, RMSE and component coverage. For rigorous confidence intervals use trial ensembles at fixed epochs; temporal samples from one realization are correlated.
- A narrow covariance is not sufficient evidence of accuracy. Validate with a full-precision orbit, realistic sensor errors, and independent higher-fidelity truth before choosing an operational model.

References: NASA's cislunar modeling reference https://ntrs.nasa.gov/citations/20220014814 and the user's existing CR3BP and lunar-surface EKF projects, documented in provenance.md.


## Long horizon forecasts

`forecast_horizons` integrates continuously through the largest requested nominal-period multiple. Requested report epochs are inserted exactly into the propagation grid but are not treated as additional observation times unless they coincide with the configured cadence. Both measurement cases use the same prior, truth, schedule and visibility conditions. Covariance updates are linearized on the nominal trajectory with zero innovations; these results quantify conditional local uncertainty and do not establish filter consistency. No orbit state or covariance resets occur at period boundaries. The report records Moon distance so geometry changes are visible rather than hidden by periodic repetition.


## Observer uncertainty contribution to Gateway tracking

For known target position rG and observer position rO, ρ = rG − rO and u = ρ/|ρ|. The derivative of line-of-sight unit direction with respect to observer position is J = −(I − uuᵀ)/|ρ|. With an orthonormal two-axis tangent basis E, the local pointing covariance is E J Prr Jᵀ Eᵀ. The reported pointing RSS σ is its trace square root in arcseconds. A numerical six-state observer Jacobian maps the full observer covariance into alpha, delta and their rates; velocity uncertainty and state cross-covariances are retained.

This mapping treats Gateway as exactly known and does not quantify Gateway orbit-estimation errors. Observer errors remain correlated over time. Camera noise may be combined at a single epoch if independent, but multi-epoch Gateway estimation requires a joint/consider or correlated-noise formulation. Visibility and radiometry of the spacecraft camera are outside this geometric uncertainty calculation.
