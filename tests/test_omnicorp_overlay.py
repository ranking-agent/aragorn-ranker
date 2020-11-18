"""Test scoring."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
# this will load all the json test files into global objects to use in a test
from .fixtures import famcov_new, property_coalesce

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

def test_omnicorp_overlay_with_set(property_coalesce):
    """Test that omnicorp_overlay() runs without errors."""
    response = client.post('/omnicorp_overlay', json={"message": property_coalesce})

    # load the json
    answer = json.loads(response.content)

    x = answer['results'][0]

    #This question is a -> set(b), so the edge bindings shoudl have one qgid "ab" and then a bunch of "s*"
    #There are 9 of them, for 10 in total

    assert(len(answer['results'][0]['edge_bindings']) == 10)
