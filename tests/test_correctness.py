"""Test correctness (publication) weighting."""
# pylint: disable=redefined-outer-name,no-name-in-module,unused-import
# ^^^ this stuff happens because of the incredible way we do pytest fixtures
import json

from fastapi.testclient import TestClient

from ranker.server import APP
from .fixtures import to_weight

client = TestClient(APP)


def test_weight(to_weight):
    """Test that weight() runs without errors."""
    response = client.post('/weight_correctness', json={
        "message": to_weight
    })
