"""Test scoring."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
from reasoner_pydantic import Response
# this will load all the json test files into global objects to use in a test
from .fixtures import omnicorp_input, property_coalesce

# start a client
client = TestClient(APP)


def test_omnicorp_overlay(omnicorp_input):
    """Test that omnicorp_overlay() runs without errors."""

    #Is the input valid
    pydantic_input = Response.parse_obj(omnicorp_input)

    response = client.post('/omnicorp_overlay', json=omnicorp_input)

    # load the json
    answer = response.json()

    # make sure the there are the same number of results that went in
    assert(len(answer['message']['results']) == len(omnicorp_input['message']['results']))

    # assert there are node bindings
    assert(len(answer['message']['results'][0]['node_bindings']) == 2)

    #assert that there are now two auxiliary graphs
    assert(len(answer['message']['auxiliary_graphs'])) == 2

    # assert that the analysis support is in auxiliary graphs
    omnicorp_graph = answer["message"]["results"][0]["analyses"][0]["support_graphs"][0]
    assert omnicorp_graph in answer["message"]["auxiliary_graphs"]

    #assert that the edges in the omnicorp support graph are the right # and the right predicate
    omnicorp_edges = answer["message"]["auxiliary_graphs"][omnicorp_graph]["edges"]
    assert len(omnicorp_edges) == 2
    assert omnicorp_edges[0]["predicate"] == "literature_co-occurrence"
    assert omnicorp_edges[1]["predicate"] == "literature_co-occurrence"

