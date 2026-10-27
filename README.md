# Predicting halo-observer delta-v with neural NARX

The goal is to predict the halo observer's **delta-v vector needed to support
Gateway tracking**, and use the data to diagnose prediction failures.

Start with [Notebook 01](notebooks/01_narx_results.ipynb). It now contains only:

1. The available delta-v targets and a check of their meaning.
2. A genuine `narxpy` network with four input and four past delta-v delays,
   predicting three delta-v components in rotating-frame m/s.
3. Held-out prediction errors, compared with the existing gradient-boosted
   baseline, zero delta-v and previous delta-v.
4. Three diagnostic figures: predicted versus target delta-v, individual burn
   errors, and the relationship between the labels and Gateway tracking error.

**The critical data limitation:** the existing STM labels correct departure from
the halo reference; they do not directly optimize Gateway tracking error. The
notebook makes this mismatch explicit. Predicting these labels cannot establish
that the predicted burns maintain Gateway tracking limits.

The focused model was actually trained for 600 epochs. On 65 held-out decisions,
recursive vector RMSE was 0.230 m/s versus 0.198 m/s for zero delta-v. On the nine
nonzero targets, RMSE was 0.514 versus 0.531 m/s, respectively. There are only
41 usable nonzero training examples, and recursive feedback increases burn error
relative to one-step prediction. These are measured limitations of this experiment,
not a claim that NARX universally fails.

## Reproduce

```sh
python -m pip install -e ".[dev,neural-narx]"
python -m pytest -q
python scripts/check_notebooks.py --one notebooks/01_narx_results.ipynb
```

The notebook reuses the saved independent training/validation/test teacher data,
or regenerates them if missing. It fits scalers and weights on training only,
selects an epoch using validation, and then evaluates test trajectories.
Prediction metrics, weights, held-out vectors and PNG figures are saved in
[artifacts/delta_v_prediction](artifacts/delta_v_prediction/).

[Notebook 00](notebooks/00_gateway_tracking_comparison.ipynb) and its original
visualizations remain unchanged. The gradient-boosted implementation remains
unchanged in `two_stage_narx.py`.

The earlier annual controller experiment remains archived under
[artifacts/neural_narx](artifacts/neural_narx/), with its original verification
record applying to commit `c6bb606`. It is not rerun by Notebook 01 and is not
used as proof of delta-v prediction accuracy. Its optional command-line entry
point remains `scripts/run_one_year_study.py`.
