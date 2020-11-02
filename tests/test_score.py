"""Test scoring."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
# this will load all the json test files into global objects to use in a test
from .fixtures import weighted2

# start a client
client = TestClient(APP)


def test_score(weighted2):
    """Test that score() runs without errors."""
    response = client.post('/score', json={"message": weighted2})

    # load the json
    answer = json.loads(response.content)

    # make sure the there are 3 results
    assert(len(answer['results']) == 3)

    # assert there are node bindings
    assert(len(answer['results'][0]['node_bindings'])== 3)

    # assert there are node bindings
    assert(len(answer['results'][0]['edge_bindings']) == 7)
