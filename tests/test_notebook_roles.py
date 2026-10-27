"""Protect the distinct purposes and visual content of the research notebooks.

Notebook 00 motivates the problem and must not lose its historical figures.
Notebook 01 is where one-year teacher datasets and learned rollouts belong.
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


def test_notebook_01_exclusively_owns_one_year_generation_and_results():
    results = nbformat.read(NB01, as_version=4)
    text = _source(results)
    for required in (
        "YEAR_DAYS = 365",
        "generate_closed_loop_dataset(",
        "fit_observer_policy(",
        "run_paired_study(",
        "observer_truth_position_departure_km",
        "gateway_tracking_position_error_km",
    ):
        assert required in text, f"Notebook 01 missing annual study: {required}"
    assert "run_gateway_comparison(" not in text
