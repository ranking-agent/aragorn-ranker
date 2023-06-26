"""Test scoring."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
from reasoner_pydantic import Response
# this will load all the json test files into global objects to use in a test
from .fixtures import omnicorp_input, property_coalesce, treatsMONDO0018912

# start a client
client = TestClient(APP)


#def test_omnicorp_overlay(omnicorp_input):
def test_omnicorp_overlay(treatsMONDO0018912):
    """Test that omnicorp_overlay() runs without errors."""
    omnicorp_input = treatsMONDO0018912

    #Is the input valid
    pydantic_input = Response.parse_obj(omnicorp_input)

    response = client.post('/omnicorp_overlay', json=omnicorp_input)
    assert response.status_code == 200

    # load the json
    answer = response.json()

    # make sure the there are the same number of results that went in
    assert(len(answer['message']['results']) == len(omnicorp_input['message']['results']))

    # There should be new edges
    assert(len(answer['message']['knowledge_graph']["edges"]) > len(omnicorp_input['message']['knowledge_graph']['edges']))

    # assert there are node bindings
    assert(len(answer['message']['results'][0]['node_bindings']) == 2)

    #assert that there are now two auxiliary graphs
    assert(len(answer['message']['auxiliary_graphs'])) == 2

    # assert that the analysis support is in auxiliary graphs
    omnicorp_graph = answer["message"]["results"][0]["analyses"][0]["support_graphs"][0]
    assert omnicorp_graph in answer["message"]["auxiliary_graphs"]

    #assert that the edges in the omnicorp support graph are the right # and the right predicate
    # there are counts for every pair in this set of 3 curies in the test data.
    omnicorp_edges = answer["message"]["auxiliary_graphs"][omnicorp_graph]["edges"]
    assert len(omnicorp_edges) == 3
    for omni_edge in omnicorp_edges:
        assert answer["message"]["knowledge_graph"]["edges"][omni_edge]["predicate"] == "biolink:occurs_together_in_literature_with"

