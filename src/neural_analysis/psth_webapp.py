"""Documented compatibility entry point for the neural-analysis webapp."""

from __future__ import annotations

import sys


if __name__ == "__main__":
    from src.neural_analysis.webapp.app import main

    main(sys.argv[1:])
else:
    from src.neural_analysis.webapp import app as _app

    sys.modules[__name__] = _app
