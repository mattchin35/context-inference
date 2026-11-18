"""Keep spawned test workers independent of earlier Streamlit AppTest scripts."""

import sys
from types import ModuleType
import pytest


@pytest.fixture(scope="module", autouse=True)
def prior_browser_test_main(tmp_path_factory):
    """Simulate a prior module leaking an AppTest __main__; restore the incoming module.

    No experimental data. The fake module names a temporary browser-only script
    that must never be used as the entry point for spawned analysis workers.
    """
    original = sys.modules["__main__"]
    fake = ModuleType("__main__")
    path = tmp_path_factory.mktemp("leaked-browser-main") / "browser.py"
    path.write_text("raise RuntimeError('Browser script is not a worker entry point')\n")
    fake.__file__ = str(path)
    sys.modules["__main__"] = fake
    yield str(path)
    sys.modules["__main__"] = original


def test_prior_browser_main_is_not_inherited(prior_browser_test_main):
    """Input leaked script path; function-level isolation restores pytest before worker tests."""
    assert sys.modules["__main__"].__file__ != prior_browser_test_main
