"""Serve the mock test sites (store + dashboard) on http://127.0.0.1:8765.

Run with:  python scripts/serve_sites.py
Then open: http://127.0.0.1:8765/store/index.html
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Mock Test Sites")

SITES_ROOT = Path(__file__).resolve().parent.parent / "sites"

app.mount(
    "/store",
    StaticFiles(directory=str(SITES_ROOT / "mock_store"), html=True),
    name="store",
)
app.mount(
    "/dashboard",
    StaticFiles(directory=str(SITES_ROOT / "mock_dashboard"), html=True),
    name="dashboard",
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")