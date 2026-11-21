"""Validate executed notebooks and scientific outputs without fixed figure counts."""
import json
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'stationkeeping'
NOTEBOOKS = (
    ROOT / 'notebooks' / '01_stationkeeping_identification.ipynb',
    ROOT / 'notebooks' / '02_stm_greybox_stationkeeping.ipynb',
)

for path in NOTEBOOKS:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert code_cells, f'{path.name}: no code cells'
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            assert ';' not in cell.source, f'{path.name}: prose contains a semicolon'
        if cell.cell_type != 'code':
            continue
        assert cell.execution_count is not None, (path.name, 'unexecuted code cell')
        for output in cell.outputs:
            assert output.output_type != 'error', (path.name, 'cell execution error')
            data = output.get('data', {})
            markdown = str(data.get('text/markdown', ''))
            if '**Purpose:**' in markdown or '**Result:**' in markdown:
                assert '**Purpose:**' in markdown and '**Result:**' in markdown
                assert ';' not in markdown, f'{path.name}: displayed prose contains a semicolon'
    images = sum(
        'image/png' in output.get('data', {})
        for cell in code_cells
        for output in cell.outputs
    )
    print(path.name, 'executed code cells:', len(code_cells), 'inline figures:', images)

figure_files = sorted((OUT / 'figures').glob('*.png'))
assert figure_files, 'No saved notebook figures found'
assert all(path.stat().st_size > 0 for path in figure_files), 'A saved figure is empty'
print('Saved figures:', len(figure_files))

config = json.loads((ROOT / 'configs' / 'stationkeeping.json').read_text())
meta = json.loads((OUT / 'data' / 'experiment.json').read_text())
assert meta['measurement_source'] == 'lso_angles'
seeds = meta['split_seeds']
assert not (set(seeds['training']) & set(seeds['validation']))
assert not (set(seeds['training']) & set(seeds['test']))
assert not (set(seeds['validation']) & set(seeds['test']))
settings = meta['settings']
assert settings['position_limit_km'] == config['position_limit_km']
assert settings['velocity_limit_m_s'] == config['velocity_limit_m_s']

fit = json.loads((OUT / 'greybox_fit.json').read_text())
assert np.isfinite(fit['gains']).all()
print('Estimated actuator gains:', fit['gains'])
print('Prediction scores:', (OUT / 'prediction_scores.json').read_text())

with np.load(OUT / 'residual_checks.npz') as checks:
    print('Ljung-Box rejections after Holm by output channel:',
          (checks['ljung_box_holm_p'] < .05).sum(axis=0).tolist())
    print('Standardized innovation mean:', checks['standardized_mean'].tolist())
print('Optical residual summary:', (OUT / 'optical_residual_summary.json').read_text())
print('Long uncontrolled propagation:', (OUT / 'uncontrolled_summary.json').read_text())
for split in ('training', 'validation', 'test'):
    with np.load(OUT / 'data' / (split + '.npz')) as record:
        assert np.isnan(record['angles'][~record['visible']]).all()
        print(split, 'visible epoch fraction:',
              float(np.any(record['visible'], axis=-1).mean()))

summary = json.loads((OUT / 'closed_loop_summary.json').read_text())
print('Observed nonlinear corridor result:', json.dumps(summary, indent=2))
assert np.isfinite(summary['maximum_position_km'])
assert np.isfinite(summary['maximum_velocity_m_s'])
annual_horizon = config.get('annual_horizon_days')
if annual_horizon is not None:
    assert summary['completed'], 'configured annual simulation did not complete'
    assert np.isclose(summary['requested_duration_days'], annual_horizon)
    assert np.isfinite(summary['annual_delta_v_m_s'])
    print('Cumulative commanded delta-v [m/s]:', summary['annual_delta_v_m_s'])
# Scientific execution success is separate from controller corridor success.
