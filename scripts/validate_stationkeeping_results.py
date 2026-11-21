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
    assert all(';' not in c.source for c in nb.cells if c.cell_type=='markdown'), name+' prose contains a semicolon'
    for cell in nb.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None, (name,'unexecuted code')
            assert not any(o.output_type=='error' for o in cell.outputs)
    figures=sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))
    expected=8 if name.startswith('01_') else 4
    assert figures == expected, (name,'unexpected figure count',figures,expected)
    captions=sum('**Purpose:**' in str(o.get('data',{}).get('text/markdown','')) and
                 '**Result:**' in str(o.get('data',{}).get('text/markdown',''))
                 for c in nb.cells for o in c.get('outputs',[]))
    assert captions == figures, (name,'each figure needs purpose/result sentences',captions,figures)
    for c in nb.cells:
        for o in c.get('outputs',[]):
            assert ';' not in str(o.get('data',{}).get('text/markdown','')), name+' displayed prose contains a semicolon'
    print(name,'executed, figures:',figures)
meta=json.loads((OUT/'data/experiment.json').read_text())
assert meta['measurement_source']=='lso_angles'
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
with np.load(OUT/'residual_checks.npz') as checks:
    print('Ljung-Box rejections after Holm by output channel:',
          (checks['ljung_box_holm_p']<.05).sum(axis=0).tolist())
    print('Standardized innovation mean:',checks['standardized_mean'].tolist())
print('Optical residual summary:',(OUT/'optical_residual_summary.json').read_text())
print('Long uncontrolled propagation:',(OUT/'uncontrolled_summary.json').read_text())
for split in ('training','validation','test'):
    with np.load(OUT/'data'/(split+'.npz')) as rec:
        assert np.isnan(rec['angles'][~rec['visible']]).all()
        print(split,'visible epoch fraction:',float(np.any(rec['visible'],axis=-1).mean()))
summary=json.loads((OUT/'closed_loop_summary.json').read_text())
print('Observed nonlinear corridor result:',json.dumps(summary,indent=2))
assert np.isfinite(summary['maximum_position_km'])
assert np.isfinite(summary['maximum_velocity_m_s'])
assert summary['completed'], 'one-year simulation did not complete'
assert np.isclose(summary['requested_duration_days'],365.25)
assert np.isfinite(summary['annual_delta_v_m_s'])
print('One-year cumulative commanded delta-v [m/s]:',summary['annual_delta_v_m_s'])
# Scientific execution success is separate from controller corridor success.

