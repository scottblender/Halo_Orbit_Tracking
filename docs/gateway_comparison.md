# One-period Gateway estimation comparison

## Two matched cases

Truth is the same nominal initial-value trajectory in both cases. There are no added observer maneuvers or physical off-nominal offsets. An imperfect prior estimate represents uncertainty in knowledge. Gateway measurements and the random Gateway initial estimation error are shared between cases. The observer and Gateway initial priors are separate independent assumptions.

The perfect case uses the exact observer state at every Gateway measurement. Its Gateway initial state remains uncertain and its camera remains noisy. The imperfect case uses a six-state observer estimate driven by LSO angles, and a six-state Gateway estimate driven by observer-to-Gateway angles. The default duration is exactly one supplied observer nominal period, 1.8572 TU.

## Measurements and observations

The angular model is h(xo,xg), computed from the relative position in model inertial axes. Its two normalized-state Jacobians are Ho and Hg. All six state coordinates are estimated, but the instantaneous angle-only measurement has direct position sensitivity; velocity information accumulates through propagation. A shared truth-based Earth/Moon segment-occultation gate supplies the same Gateway schedule to both estimators. The LSO gate uses the configured spherical elevation mask. No additional observation is inserted at the final off-cadence reporting time.

## Correlation-aware imperfect-observer filter

The imperfect case carries x=[xo,xg] and the full block covariance:

```text
P = [ Poo  Pog ]
    [ Pgo  Pgg ]
```

The state-transition matrix is block diagonal because the spacecraft do not dynamically interact. The cross-covariance propagates as Pog(t+) = Fo Pog(t) Fgᵀ.

For LSO data H=[Hlso,0], the full Kalman gain is used. This updates the observer from the LSO and can correct Gateway through its correlation with observer errors. The observer's own marginal estimate is identical to an LSO-only filter: Gateway data never improve it.

For Gateway data H=[Ho,Hg], the ordinary gain is computed, then its observer rows are set to zero. This is a Schmidt/consider-style update: Gateway is active and the observer is a considered uncertain state. Joseph covariance updates preserve the cross-correlation. This accounts for the temporal correlation of observer errors rather than adding Ho Poo Hoᵀ as independent noise at every epoch.

The perfect case is a six-state Gateway filter with Ho absent. The zero-observer-covariance limit of the imperfect forecast is checked against that perfect case. The default baseline uses zero process noise for both spacecraft; if configured, observer process noise enters only its block.

## Actual errors and forecast covariance

Two kinds of output are intentionally separated:

- Noisy state estimation, with a seeded initial error and one shared Gateway noise realization. Actual errors compare the estimates to simulated truth. Arc RMSE is a time average for this one realization, not a Monte Carlo ensemble statistic.
- Nominal covariance forecasting, with zero innovations and Jacobians evaluated along nominal truth. It predicts local conditional covariance. It does not validate actual error bounds.

Final RSS sigma is sqrt(trace(P_position)) or sqrt(trace(P_velocity)) after conversion to model Moon-centered inertial units. It is not a one-axis standard deviation or a guaranteed three-dimensional confidence radius.

The earlier observer EKF consistency issue is retained as a limitation rather than hidden by altering sensor noise or P0. Main numerical results are conditional on this CR3BP model, orbital phase, Gateway prior, cadence and observing gates. A calibrated operational accuracy claim requires improved nonlinear filtering and appropriate validation.
