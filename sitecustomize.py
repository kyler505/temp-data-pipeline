"""Repository-local sitecustomize.

Purpose: make `src/` importable without requiring an editable install.

This helps on systems (and some test runners) where editable installs / .pth
processing is unreliable. It is a no-op if `src/` is already on sys.path.

Python auto-imports `sitecustomize` at startup if it is importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
src = repo_root / "src"

if src.exists():
    src_str = str(src)
    if src_str not in sys.path:
        # Prepend so local code wins over any globally installed packages.
        sys.path.insert(0, src_str)
