"""Test scoring."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
# this will load all the json test files into global objects to use in a test
from .fixtures import famcov_new

# start a client
client = TestClient(APP)


def test_omnicorp_overlay(famcov_new):
    """Test that omnicorp_overlay() runs without errors."""
    response = client.post('/omnicorp_overlay', json={"message": famcov_new})

    # load the json
    answer = json.loads(response.content)

    # make sure the there are 27 results
    assert(len(answer['results']) == 27)

    # assert there are node bindings
    assert(len(answer['results'][0]['node_bindings'])== 3)

    # assert there are node bindings
    assert(len(answer['results'][0]['edge_bindings']) == 9)
