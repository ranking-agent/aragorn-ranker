"""Test correctness (publication) weighting."""
# pylint: disable=redefined-outer-name,no-name-in-module,unused-import
# ^^^ this stuff happens because of the incredible way we do pytest fixtures
import json

from fastapi.testclient import TestClient

from ranker.server import APP
from .fixtures import to_weight

client = TestClient(APP)


def test_weight(to_weight):
    """Test that weight() runs without errors and that the weights are correctly ordered."""
    response = client.post('/weight_correctness', json=to_weight)

    weightresponse = response.json()
    #The input is constructed to have a series of 4 edges

    weights = {}
    # weights = { e['kg_id']: e['weight'] for e in ebs}

    ebs = weightresponse['message']['results'][0]['edge_bindings']

    for e in ebs:
        weights[ebs[e][0]['id']] = ebs[e][0]['attributes'][0]['value']

    #there are 3 pubs in the malformed array, and a pubcount of 2
    assert weights['badpublicationsarray'] > weights['correctpublicationscount']
    #good publications array also has 2
    assert weights['correctpublicationsarray'] == weights['correctpublicationscount']
    assert weights['correctpublicationsarray'] > weights['emptypublicationsarray']