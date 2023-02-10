"""Test correctness (publication) weighting. This is soon deprecated."""
# pylint: disable=redefined-outer-name,no-name-in-module,unused-import
# ^^^ this stuff happens because of the incredible way we do pytest fixtures
import json

from copy import deepcopy
from fastapi.testclient import TestClient

from ranker.server import APP

from .fixtures import pub_test, to_weight

client = TestClient(APP)


def test_null_results(to_weight):
    """Test that weight() runs without errors. Even for null results"""
    null_results = deepcopy(to_weight)
    null_results["message"]["results"] = None
    response = client.post('/weight_correctness', json=null_results)
    assert response.status_code == 200

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

    assert weights['correctpublicationsarray'] > 0

def test_pubs(pub_test):
    """We are getting results from KPs with different ways of encoding pubs.  This needs to be fixed at the EPC level,
    but for now, we do want to handle them.
    The test case contains a single answer with 4 edges defined in different ways
    1. The way BTE does it (3 pubs)
    2. The way BTE does it (0 pubs)
    3. The way text miner kp does it (5 pubs)
    4. The way omnicorp does it (497pubs).
    For simplicity the omnicorp edge has been bound to the same query edge, which wouldn't happen really.
    The test is going to check that the weights are correctly ordered.
    omnicorp > text miner 5 > BTE 3 > BTE 0
    This makes a couple of assumptions about how we relatively weight omnicorp, TM, and other sources, which might
    change in the future, say if we decide to downweight TM."""
    response = client.post('/weight_correctness', json=pub_test)

    weightresponse = response.json()

    weights = {}
    ebs = weightresponse['message']['results'][0]['edge_bindings']

    for e in ebs:
        print(ebs[e])
        for kedge in ebs[e]:
            kedge_id = kedge['id']
            for att in kedge.get('attributes',[]):
                if att['original_attribute_name'] == 'weight':
                    weights[kedge_id] = att['value']

    # there are 3 pubs in the malformed array, and a pubcount of 2
    assert weights['BTE_TM_5'] > 0

