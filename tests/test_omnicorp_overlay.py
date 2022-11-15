"""Test scoring."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
# this will load all the json test files into global objects to use in a test
from .fixtures import omnicorp_input, property_coalesce

# start a client
client = TestClient(APP)


def test_omnicorp_overlay(omnicorp_input):
    """Test that omnicorp_overlay() runs without errors."""
    response = client.post('/omnicorp_overlay', json=omnicorp_input)

    # load the json
    answer = response.json()

    # make sure the there are the same number of results that went in
    assert(len(answer['message']['results']) == len(omnicorp_input['message']['results']))

    # assert there are node bindings
    assert(len(answer['message']['results'][0]['node_bindings']) == 2)

    # assert there are 2 edge bindings.  The original one, and the one that came out of omnicorp
    assert(len(answer['message']['results'][0]['edge_bindings']) == 1)

def test_omnicorp_overlay_with_set(property_coalesce):
    """Test that omnicorp_overlay() runs without errors."""
    response = client.post('/omnicorp_overlay', json=property_coalesce)
    assert response.status_code==200