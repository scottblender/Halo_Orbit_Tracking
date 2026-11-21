"""Lightweight checks for generated station-keeping notebooks and outputs."""
import json
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / 'notebooks'
OUT = ROOT / 'artifacts' / 'stationkeeping'

notebooks = sorted(NOTEBOOK_DIR.glob('*.ipynb'))
assert notebooks, 'No notebooks found'
for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert code_cells, f'{path.name}: no code cells'
    for cell in code_cells:
        assert cell.execution_count is not None, f'{path.name}: code cell was not executed'
        assert not any(output.output_type == 'error' for output in cell.outputs), (
            f'{path.name}: cell execution error'
        )
    inline_figures = sum(
        'image/png' in output.get('data', {})
        for cell in code_cells
        for output in cell.outputs
    )
    print(path.name, 'executed code cells:', len(code_cells),
          'inline figures:', inline_figures)

figure_dir = OUT / 'figures'
figures = sorted(figure_dir.glob('*.png')) if figure_dir.exists() else []
assert figures, 'No saved figures found'
assert all(path.stat().st_size > 0 for path in figures), 'A saved figure is empty'
print('Saved figures:', len(figures))

summary_path = OUT / 'closed_loop_summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    for key, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert np.isfinite(value), f'{key} is not finite'
    print('Closed-loop summary loaded. Cumulative commanded delta-v:',
          summary.get('annual_delta_v_m_s', summary.get('total_commanded_delta_v_m_s')))
