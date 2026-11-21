"""Check executed artifacts, without changing or concealing corridor failures."""
import json
from pathlib import Path
import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stationkeeping'
for name in ('01_stationkeeping_identification.ipynb','02_stm_greybox_stationkeeping.ipynb'):
    nb=nbformat.read(ROOT/'notebooks'/name,as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None, (name,'unexecuted code')
            assert not any(o.output_type=='error' for o in cell.outputs)
    figures=sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))
    assert figures >= 5, (name,'missing figures',figures)
    print(name,'executed, figures:',figures)
meta=json.loads((OUT/'data/experiment.json').read_text())
seeds=meta['split_seeds']
assert not (set(seeds['training'])&set(seeds['validation']) or
            set(seeds['training'])&set(seeds['test']) or
            set(seeds['validation'])&set(seeds['test']))
assert meta['settings']['position_limit_km']==10
assert meta['settings']['velocity_limit_m_s']==.1
fit=json.loads((OUT/'greybox_fit.json').read_text())
assert np.isfinite(fit['gains']).all()
print('Estimated actuator gains:',fit['gains'])
print('Prediction scores:',(OUT/'prediction_scores.json').read_text())
summary=json.loads((OUT/'closed_loop_summary.json').read_text())
print('Observed nonlinear corridor result:',json.dumps(summary,indent=2))
assert np.isfinite(summary['maximum_position_km'])
assert np.isfinite(summary['maximum_velocity_m_s'])
# Scientific execution success is separate from controller corridor success.
