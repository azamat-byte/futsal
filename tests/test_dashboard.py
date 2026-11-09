from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_dashboard_overview():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard/app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Your match analysis workspace"


def test_dataset_page():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard/app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Dataset studio").run()
    assert not app.exception
    assert app.title[0].value == "Dataset studio"


@pytest.mark.parametrize(
    "page", ["Match viewer", "Analytics", "Calibration", "Identities & events", "Exports", "Evaluation"]
)
def test_dashboard_pages_have_no_errors(page):
    app = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard/app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error


def test_match_viewer_completed_run_window():
    root = Path(__file__).parents[1]
    from futsal.jobs import list_jobs

    jobs = list_jobs(root / "runs")
    target = next((j for j in jobs if j["id"] == "f36dcb388903"), None)
    if target is None or target["state"] != "completed":
        pytest.skip("User's completed videoplayback run is not available")
    app = AppTest.from_file(str(root / "dashboard/app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Match viewer").run()
    app.selectbox[0].set_value(target).run()
    assert not app.exception
    assert not app.error
    assert any(slider.label == "Inspect source timestamp" for slider in app.slider)
