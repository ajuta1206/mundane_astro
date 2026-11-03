from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_runs_without_exceptions():
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception, at.exception
