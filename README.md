# Halo station-keeping delta-v: polynomial NARX versus STM

[Notebook 01](notebooks/01_narx_results.ipynb) now uses one **sparse quadratic polynomial NARX**, replacing the neural network in the active experiment. It predicts three rotating-frame delta-v components [m/s] **given the STM reference burn schedule**. Burn/no-burn and timing are supplied, not learned. Gateway tracking remains a separate post-processing question.

Inputs are rotating position/velocity departure [km, m/s], sine/cosine of orbital phase, and the supplied waiting time. Four input samples (current plus three previous) and four previous issued delta-v commands form a causal NARX. The frame conversion removes transport velocity from stored inertial departure. Training and deployment use the same conversion.

Orthogonal matching pursuit fits a sparse subset of 1,080 possible linear/quadratic terms. Separate trajectories are used for training (20), validation (5), and test (5). Scaling and coefficients use training burns only; validation selects 1, 2, 4, 8, or 12 terms per component. Test data do not select the model. Saved coefficients describe standardized coordinates.

The notebook shows three figures: predicted delta-v versus STM, prediction residuals, and nonlinear orbit departure with cumulative delta-v. The fitted model, histories, coefficients, data audit, metrics and termination reasons are saved in [artifacts/polynomial_narx](artifacts/polynomial_narx/).

## STM reference and the orbit test

STM means **state transition matrix**. The reference procedure checks estimated halo departure eight times per orbit. At 50 km position departure or 0.5 m/s velocity departure, it tries burns now or 22.5, 45, or 67.5 degrees later in the time-based orbit cycle. It solves delta-v using the STM, verifies improvement by nonlinear propagation, rejects impulses above 10 m/s, and selects the lowest departure cost with a small fuel penalty. No additional burn is scheduled while one is pending. These are calculated reference commands, not unique truth.

The planned 365-day comparison first computes the STM schedule, then supplies it to the polynomial run. Both use identical initial states and epoch-indexed measurement noise. For the first four decisions, both use the same STM commands to initialize history. Thereafter NARX uses its own estimated state and previous issued commands, with no reference delta-v replacement. Measurements occur every six hours plus decision epochs.

**This tests delta-v prediction conditional on known burn times, not autonomous station keeping.** The reference schedule can become inappropriate as the NARX trajectory diverges. Curves stop at a decision-sampled 20,000 km departure cutoff or numerical failure; sampled departure can overshoot the cutoff. Missing future states are not filled in. This is one paired orbit test, not a Monte Carlo success-rate estimate.

## Reproduce

```sh
python -m pip install -e ".[dev,polynomial-narx]"
python -m pytest -q --basetemp=work/pytest-local
python scripts/check_notebooks.py --one notebooks/01_narx_results.ipynb
```

The existing STM dataset is reused from its historical `artifacts/neural_narx/teacher_training` location, or generated if missing. That folder contains reference trajectories, not neural predictions. The audit replays targets and checks units, histories, configuration and cadence. PyTorch and narxpy are not required by the active polynomial experiment; the optional `neural-narx` dependencies support historical experiments/tests.

[Notebook 00](notebooks/00_gateway_tracking_comparison.ipynb) is unchanged by this replacement. Its earlier visualizations and full-year STM tracking, RMSE and uncertainty study remain intact. Earlier neural implementations and the gradient-boosted baseline remain available as historical source/artifacts; Notebook 01 does not run or compare them.

```sh
python scripts/check_notebooks.py --one notebooks/00_gateway_tracking_comparison.ipynb --from-cell 26
```
