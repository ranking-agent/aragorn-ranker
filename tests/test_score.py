"""Test scoring."""
import json
from fastapi.testclient import TestClient
from ranker.server import APP
# this will load all the json test files into global objects to use in a test
from .fixtures import score_test, weighted2, weighted_answer, schizo, treatsSchizophreniaw, weighted_set
from pytest import approx

# start a client
client = TestClient(APP)


#this test is out of date because its input does not match the format of weight correctness.
# the test data has weight in the edge bindings, rather than in the attribute of the edge bindings.
def xtest_nonzero_score(weighted_answer):
    """Test that score() runs without errors."""
    response = client.post('/score', json=weighted_answer)

    # load the json
    resp = response.json()

    answer = resp['message']['results']

    # make sure the there are 3 results
    assert(len(answer) == 1)

    assert(answer[0]['score'] > 0)

#this test is out of date because its input does not match the format of weight correctness.
# the test data has weight in the edge bindings, rather than in the attribute of the edge bindings.
def test_score(weighted2):
    """Test that score() runs without errors."""
    response = client.post('/score', json=weighted2)

    # load the json
    resp = response.json()

    answer = resp['message']['results']

    # make sure the there are 3 results
    assert(len(answer) == 3)

    # assert there are node bindings
    assert(len(answer[0]['node_bindings']) == 3)

    # assert there are node bindings
    assert(len(answer[0]['edge_bindings']) == 5)

def test_score_schizo(schizo):
    """Test that score() runs without errors."""
    response = client.post('/score', json=schizo)

    # load the json
    resp = response.json()

    answer = resp['message']['results']

    assert response.status_code == 200

def test_score_schizo(treatsSchizophreniaw):
    """Test that score() runs without errors."""
    response = client.post('/score', json=treatsSchizophreniaw)

    # load the json
    resp = response.json()

    answer = resp['message']['results']

    assert response.status_code == 200

def test_score_set(weighted_set):
    response = client.post('/score', json=weighted_set)
    resp = response.json()
    answer = resp['message']['results']
    print(answer[0]['score'])
    assert response.status_code == 200

def test_score_set_symmetry(weighted_set):
    """For our sets, it should not matter the order in which values are applied to edges"""
    edge_bindings = weighted_set['message']['results'][0]['edge_bindings']['e01']
    #Set the weights one way, score it
    edge_bindings[0]['attributes'][0]['value'] = 2
    edge_bindings[1]['attributes'][0]['value'] = 1
    response = client.post('/score', json=weighted_set)
    resp = response.json()
    answer = resp['message']['results']
    score_one = answer[0]['score']
    #Reverse the weights, score it
    edge_bindings[0]['attributes'][0]['value'] = 1
    edge_bindings[1]['attributes'][0]['value'] = 2
    response = client.post('/score', json=weighted_set)
    resp = response.json()
    answer = resp['message']['results']
    score_two = answer[0]['score']
    assert score_one == approx(score_two)


def test_basic_scoring(score_test):
    response = client.post('/score', json=score_test)
    resp = response.json()
    answer = resp['message']['results']
    assert response.status_code == 200
    assert len(answer) == 2
    assert answer[0]['score'] == approx(0.2884698783406143)
    assert answer[1]['score'] == approx(0.2884698783406143)
