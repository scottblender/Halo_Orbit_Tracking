# Halo burns: one NARX prediction and one STM reference

[Notebook 01](notebooks/01_narx_results.ipynb) asks two questions:

- Does one `narxpy` network predict the reference delta-v and burn phase?
- What happens to the halo orbit when those predicted burns are applied?

It contains three plots: **NARX versus STM burns**, **prediction residuals**
(NARX minus STM), and **orbit departure after applying each burn sequence**.
A short statistics table counts missed burns and measures delta-v and timing
errors. A phase-count table shows which burn timings the training data cover.
No other prediction models are compared in this notebook. Gateway tracking is
left for later post-processing.

## What STM and the burn procedure mean

STM means **state transition matrix**. It estimates how a small change now
changes the future orbit. The STM reference calculation uses that information
to calculate a correction; it is not a neural network or a perfect answer.

The existing procedure checks the estimated halo departure eight times per
orbit. At **50 km position departure or 0.5 m/s velocity departure**, it tries
burning now or 22.5, 45, or 67.5 degrees later in the time-based orbit cycle.
It calculates a delta-v for each option, propagates the nonlinear orbit to
check it, rejects impulses above 10 m/s or corrections that do not improve
departure, and chooses the best remaining correction with a small fuel penalty.
No additional burn is scheduled while one is pending. This choose-and-apply
procedure is what the code calls a controller.

The single NARX learns the burn/no-burn choice, three delta-v components, and
waiting-time scores together. It has four input-history steps, four past-output
steps, and 64 hidden units. During its orbit test it uses its own predictions,
with no STM replacement burns.

## Reproduce

```sh
python -m pip install -e ".[dev,neural-narx]"
python -m pytest -q
python scripts/check_notebooks.py --one notebooks/01_narx_results.ipynb
```

The notebook reuses independent 20/5/5 training/validation/test reference
trajectories, or regenerates them if missing. It trains one network for 600
epochs and selects the epoch using validation. Test trajectories are used only
for evaluation. It then runs one matched STM/NARX nonlinear simulation pair
with a planned 365-day span. Initial states and sensor noise are identical.
This is one paired orbit test, not a population success-rate estimate.

Results, weights, figures and underlying arrays are saved in
[artifacts/narx_stm_comparison](artifacts/narx_stm_comparison/). Curves stop on
termination; missing future states are not filled in. Completing the year does
not imply remaining inside the 100 km / 1 m/s comparison corridor.

[Notebook 00](notebooks/00_gateway_tracking_comparison.ipynb) and its original
visualizations remain unchanged. Earlier baselines, pure delta-v experiments,
and annual comparisons remain available in their existing source modules and
artifact folders; Notebook 01 does not train or compare those additional models.
