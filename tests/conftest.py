import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # import project modules from tests/


import pytest


@pytest.fixture(autouse=True)
def _forget_paused_gemini_models():
    """Each test starts with every Gemini model available."""
    import gemini_detector

    gemini_detector._paused_until.clear()
    yield
    gemini_detector._paused_until.clear()
