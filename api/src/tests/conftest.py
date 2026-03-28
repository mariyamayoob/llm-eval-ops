from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
API_SRC = ROOT / "api" / "src"
if str(API_SRC) not in sys.path:
    sys.path.insert(0, str(API_SRC))

from main import create_app


@pytest.fixture()
def client():
    workspace_tmp = ROOT / "test_artifacts" / str(uuid.uuid4())
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    db_path = workspace_tmp / "policy_desk.db"
    app = create_app(db_path=str(db_path))
    try:
        yield TestClient(app)
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)
