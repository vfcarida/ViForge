"""
Headless automated tests for ViForge Streamlit Studio using streamlit.testing.v1.AppTest.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = str(Path("src/viforge/ui/app.py").resolve())


@pytest.mark.unit
def test_streamlit_app_renders_without_exceptions():
    """Verify that the full 8-tab Streamlit Studio application renders cleanly without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run(timeout=60)

    # Verify no unhandled exceptions in the entire execution run
    assert not at.exception, f"App execution raised exception: {at.exception}"

    # Verify sidebar elements
    assert len(at.sidebar.selectbox) >= 2
    # Verify model and domain selector labels
    assert at.sidebar.selectbox[0].label == "Model Family"
    assert at.sidebar.selectbox[1].label == "Domain Specialization"

    # Verify header renders
    assert any("ViForge Specialization & Pareto Studio" in md.value for md in at.markdown)


@pytest.mark.unit
def test_streamlit_app_interactive_model_switching():
    """Verify that switching models in the sidebar updates state and reruns without error."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run(timeout=60)
    assert not at.exception

    # Select Qwen model
    at.sidebar.selectbox[0].select("Qwen/Qwen2.5-Coder-7B")
    # Select cybersecurity domain
    at.sidebar.selectbox[1].select("cybersecurity")
    at.run(timeout=60)

    assert not at.exception
    assert at.sidebar.selectbox[0].value == "Qwen/Qwen2.5-Coder-7B"
    assert at.sidebar.selectbox[1].value == "cybersecurity"


@pytest.mark.unit
def test_streamlit_app_kpi_metrics_rendered():
    """Verify that key performance indicators (metrics) render correctly."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run(timeout=60)
    assert not at.exception

    # Verify key metrics exist on overview tab
    metric_labels = [m.label for m in at.metric]
    assert "Domain Capability Gain" in metric_labels
    assert "General Retention Delta" in metric_labels
    assert "Peak VRAM Reduction" in metric_labels
    assert "Inference Latency Speedup" in metric_labels
    assert "Training Cost" in metric_labels
