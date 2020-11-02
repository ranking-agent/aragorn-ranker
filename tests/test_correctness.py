"""Test correctness (publication) weighting."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
# this will load all the json test files into global objects to use in a test
from .fixtures import to_weight

client = TestClient(APP)


def test_weight(to_weight):
    """Test that correctness() runs without errors."""
    response = client.post('/weight_correctness', json={"message": to_weight})

    # load the json
    answer = json.loads(response.content)

    # make sure the there are 3 results
    assert (len(answer['results']) == 3)

    # assert there are node bindings
    assert (len(answer['results'][0]['node_bindings']) == 3)

    # assert there are node bindings
    assert (len(answer['results'][0]['edge_bindings']) == 7)
