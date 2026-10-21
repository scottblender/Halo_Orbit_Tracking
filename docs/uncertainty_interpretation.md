# Why the current observer EKF covariance is overconfident

P0 should represent actual prior knowledge, not be tuned to achieve a preferred NEES. The retained per-axis standard deviations are 1 km position and 0.01 m/s velocity; the LSO angular-noise standard deviation remains 1 arcsecond.

A 100-sample diagnostic using the baseline input state, zero process noise and model-matched truth found:

| Diagnostic | Mean six-state NEES |
| --- | ---: |
| Prior propagation to the first observation, with covariance linearized along each estimated state | 8.04 |
| Immediately after the first angles-only update at 1 arcsec | 61.56 |
| Sensitivity check: the same first update with 5 arcsec | 10.24 |

The first visible sample is at 3.22917 days, near the closest sampled lunar passage. An additional prior-only propagation diagnostic over the complete nominal period gave mean NEES 5.80. The completed noisy 100-trial observer-filter validation gave final mean NEES about 97–101. The first-update diagnostic uses a separate fixed sample/noise stream, so its scores are not exact intermediate values from that full run.

These checks isolate a large consistency loss to measurement assimilation, rather than demonstrating that the initial covariance is too small. They strongly support nonlinear update/linearization effects: after a long gap and close passage, a very precise angular observation sharply constrains transverse position, while appreciable uncertainty remains along the line of sight. A single linearized Gaussian update can report a narrow covariance even when second-order terms leave meaningful state error. Truth and filter use the same CR3BP, so omitted physical forces cannot explain this baseline's mismatch.

This does not prove every implementation detail is correct. Dynamics/frame/measurement checks pass, but nonlinear-filter comparisons remain needed. Appropriate next experiments are an iterated EKF or UKF with the same P0 and sensor noise, together with innovation, bias, NEES and coverage checks. Do not reduce P0 or increase sensor noise solely to improve a consistency metric.

Nominal covariance propagation remains useful for studying trends. Its downstream Gateway pointing projection inherits the same conditional/local interpretation and must not yet be treated as a validated physical accuracy bound. Monte Carlo is optional validation of that approximation, not the main uncertainty-propagation method.
