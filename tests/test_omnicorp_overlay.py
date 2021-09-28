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
    assert(len(answer['message']['results'][0]['edge_bindings']) == 2)

def test_omnicorp_overlay_with_set(property_coalesce):
    """Test that omnicorp_overlay() runs without errors."""
    response = client.post('/omnicorp_overlay', json=property_coalesce)

    # load the json
    answer = json.loads(response.content)

    x = answer['message']['results'][0]

    #This question is a -> set(b), so the edge bindings shoudl have one qgid "ab" and then a bunch of "s*"
    #There are 9 of them, for 10 in total

    assert(len(answer['message']['results'][0]['edge_bindings']) == 1)
    assert(len(answer['message']['results'][0]['edge_bindings']['ab']) == 10)
