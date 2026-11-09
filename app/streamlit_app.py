"""Compatibility entry point: streamlit run app/streamlit_app.py."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), run_name="__main__")
