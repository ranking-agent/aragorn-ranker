"""Test scoring."""
# pylint: disable=redefined-outer-name,no-name-in-module,unused-import
# ^^^ this stuff happens because of the incredible way we do pytest fixtures
from fastapi.testclient import TestClient

from ranker.server import APP
from .fixtures import weighted2

client = TestClient(APP)


def test_score(weighted2):
    """Test that score() runs without errors."""
    response = client.post('/score', json={"message": weighted2})

    print(response)
