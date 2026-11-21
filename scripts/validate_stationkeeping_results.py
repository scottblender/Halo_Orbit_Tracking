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

summary = json.loads((OUT / 'closed_loop_summary.json').read_text(encoding='utf-8'))
for key, value in summary.items():
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        assert np.isfinite(value), f'{key} is not finite'

with np.load(OUT / 'closed_loop.npz') as trial:
    days = trial['day']
    burn_days = trial['burn_day']
    commands = trial['commands']
    assert days.ndim == 1 and days.size, 'No propagated epochs'
    assert np.isfinite(days).all(), 'Nonfinite propagation epochs'
    time_tolerance = 4 * np.spacing(np.max(np.abs(days)))
    assert np.all(np.diff(days) >= -time_tolerance), 'Propagation epochs run backward'
    assert np.isfinite(trial['output']).all(), 'Nonfinite propagated state'
    assert trial['output'].shape == (days.size, 6), 'State history does not match epochs'
    assert commands.shape == (burn_days.size, 3), 'Impulse history does not match burn epochs'
    assert np.isfinite(commands).all(), 'Nonfinite commanded impulse'
    assert np.all((burn_days >= days[0]) & (burn_days <= days[-1])), 'Burn outside propagation'
    total = np.linalg.norm(commands, axis=1).sum()

assert np.isclose(days[-1], summary['last_day']), 'Summary and history end on different days'
assert commands.shape[0] == summary['burns'], 'Summary and history disagree on burn count'
assert np.isclose(total, summary['total_commanded_delta_v_m_s']), 'Inconsistent cumulative delta-v'
if summary['completed']:
    assert np.isclose(days[-1], summary['requested_duration_days'], rtol=0, atol=1e-8), (
        'Completed propagation ended outside the requested horizon'
    )
annual = summary.get('annual_delta_v_m_s')
if annual is not None:
    assert summary['completed'], 'Incomplete propagation cannot report annual delta-v'
    assert np.isclose(annual, total), 'Annual and cumulative delta-v disagree'
print(f'Closed-loop propagation completed: {summary["completed"]}. Commanded delta-v: {total:.6g} m/s')
