"""Protect the distinct purposes and visual content of the research notebooks.

Notebook 00 motivates the problem and must not lose its historical figures.
Notebook 01 focuses on halo-observer delta-v prediction and data diagnostics.
"""
from pathlib import Path
import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB00 = ROOT / "notebooks" / "00_gateway_tracking_comparison.ipynb"
NB01 = ROOT / "notebooks" / "01_narx_results.ipynb"


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def test_notebook_00_keeps_original_motivation_and_visualizations():
    setup = nbformat.read(NB00, as_version=4)
    text = _source(setup)
    required_topics = (
        "Three lunar observatories",
        "Effect of observer uncertainty",
        "Stability of the corrected nominal",
        "Growth of the uncontrolled halo departure",
        "without new measurements",
        "One-year nominal reference and LSO geometry",
    )
    for topic in required_topics:
        assert topic in text, f"Notebook 00 lost setup content: {topic}"
    assert len(setup.cells) >= 26
    figures = sum(
        "image/png" in output.get("data", {})
        for cell in setup.cells for output in cell.get("outputs", [])
    )
    assert figures >= 11, "Notebook 00 must retain its original embedded plots"
    assert "generate_closed_loop_dataset(" not in text
    assert "run_paired_study(" not in text


def test_notebook_01_focuses_on_delta_v_prediction():
    results = nbformat.read(NB01, as_version=4)
    text = _source(results)
    for required in (
        "fit_delta_v_predictor(",
        "d_y=3",
        "Nonzero-target vector RMSE",
        "gateway_tracking_position_error_km",
        "**not** optimize Gateway tracking error",
    ):
        assert required in text, f"Notebook 01 missing prediction content: {required}"
    assert "run_paired_study(" not in text
    assert "run_gateway_comparison(" not in text
