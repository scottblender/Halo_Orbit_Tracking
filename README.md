# Gateway tracking with an uncertain halo observer

**Open [`notebooks/00_gateway_tracking_comparison.ipynb`](notebooks/00_gateway_tracking_comparison.ipynb).** This is the main notebook. It compares actual Lunar Gateway orbit estimates over **one halo-observer nominal period**, about 8.06 days.

| Case | Observer state used to estimate Gateway |
|---|---|
| Perfect knowledge | Exact true observer position and velocity |
| LSO-estimated knowledge | Observer state estimated from lunar-surface optical observations, carrying its uncertainty into the Gateway estimator |

Both cases have the same true trajectories, Gateway prior, Gateway camera measurements and noise realization. The experiment isolates the effect of observer knowledge. Perfect knowledge of the observer does not imply noise-free measurements or exact knowledge of Gateway.

## Run the main notebook

From the repository root on Windows, activation is unnecessary:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name halo-tracking --display-name "Python (halo tracking)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Select **Python (halo tracking)** and run the main notebook from top to bottom. If the environment already exists, skip its creation. After pulling updates, restart the notebook kernel so Python loads the updated modules.

The notebook contains:

1. A short, explicit assumptions table.
2. One-period trajectory geometry and LSO observer knowledge.
3. Gateway **actual estimation errors**, shown separately from **predicted uncertainty**.
4. A numerical comparison and automatically generated conclusions.

Generated data, configuration snapshots and figure exports go to `results/gateway_comparison/`. To reproduce them without Jupyter, run `.\.venv\Scripts\python.exe scripts/run_gateway_comparison.py`.

## Baseline assumptions

| Parameter | Value |
|---|---|
| Observer initial position / velocity standard deviation | 1 km / 0.01 m/s per axis |
| Gateway initial position / velocity standard deviation | Separately assumed 1 km / 0.01 m/s per axis |
| Angular standard deviation on each optical link | 1 arcsecond per angle |
| Measurement cadence | 30 minutes |
| LSO site | 30° N, 270° east longitude |
| LSO elevation mask | 10° |
| Main comparison measurements | Angles only |
| Dynamics | Earth–Moon CR3BP, perfect-model baseline, no maneuvers |

Edit `configs/baseline.json`. Gateway prior and camera noise have separate fields under `comparison`; the observer prior is under `estimation`. Initial cross-covariances are zero. Position and velocity uncertainties are converted to the normalized propagation coordinates.

Gateway camera availability includes Earth/Moon occultation. Sun exclusion, brightness, attitude, station-location errors and spacecraft maneuvers are not included. The LSO observing window is approximately days 3.23–4.15 for the supplied phase.

## Baseline result and interpretation

| Gateway metric | Perfect observer | LSO-estimated observer |
|---|---:|---:|
| Final predicted position RSS σ | 33.1 m | 513.2 m |
| Final actual position error, one paired realization | 21.2 m | 1528.9 m |
| Position RMSE over the arc, one paired realization | 443.9 m | 3525.7 m |

The local forecast predicts **15.5× larger final position RSS uncertainty** when the observer state is estimated from the LSO. After the LSO window ends, observer uncertainty grows again and affects Gateway tracking.

The actual errors are one paired realization, not ensemble-average performance. The earlier observer-filter validation found covariance overconfidence, so the predicted magnitudes remain **conditional local forecasts**, not validated physical accuracy guarantees. The main notebook states this limitation beside the results.

## How observer uncertainty is carried into Gateway estimation

The imperfect case propagates observer and Gateway state covariances **and their cross-covariance**. LSO observations update the observer; Gateway observations update Gateway without refining the observer. Retaining the correlation prevents repeated observer errors from being treated as independent white noise. LSO updates can also improve Gateway indirectly through that correlation. See [`docs/gateway_comparison.md`](docs/gateway_comparison.md).

## Orbit inputs and reference frames

The supplied observer row is retained unchanged:

```text
241  6  SNRHL2  T=1.8572 TU  stability=1.6875
[x,y,z,vx,vy,vz] = [1.0444,-0.0277,-0.1879,-0.0329,-0.1430,0.0810]
```

`SNRHL2` identifies a southern NRHO in the previous project, despite the initial discussion of a northern observer. The four-decimal input misses one-period closure by about 99 km; this is an initial-value study, not a validated periodic orbit or stationkeeping simulation. The earlier 5/10/20-period exploration is archived and excluded from the main workflow.

Gateway's normalized state is sourced from the earlier cislunar project's `scripts/build_target_case_database.m`. Both initial phases are defined at model t=0. The study uses μ=0.01215058560962404, LU=384400 km and TU=375190.25852 s for both trajectories. The older Gateway source uses TU=375695 s; its normalized state is used with this study's explicitly retained time scale. No epoch-specific mission trajectory is asserted.

Propagation uses Earth–Moon barycentric rotating coordinates. Physical errors and covariance summaries use model Moon-centered inertial axes, including the rotational velocity terms. See [`docs/model.md`](docs/model.md).

## Optional material

- `notebooks/advanced/`: individual trajectory, measurement, observer-filter and optional 100-trial validation diagnostics. These are not required to interpret the main comparison.
- `notebooks/archive/`: earlier long-span covariance/pointing exploration, excluded from normal execution.
- `docs/uncertainty_interpretation.md`: first-update consistency diagnostic.
- `docs/provenance.md`: prior-code references.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/check_notebooks.py
# Also run optional diagnostics:
.\.venv\Scripts\python.exe scripts/check_notebooks.py --include-advanced
```

For socket-restricted environments, append `--in-process` to execute notebooks in isolated IPython processes. `style_3d_axes` provides plot-relative label spacing that responds to camera changes, resizing and export resolution; the trajectory renderer uses compact unit scaling and equal physical axis scales.
