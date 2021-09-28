import json
from fastapi.testclient import TestClient
from ranker.server import APP
from .fixtures import omnicorp_input

client = TestClient(APP)

def test_service(omnicorp_input):
    """cascade through all ranker end points."""
    response = client.post('/omnicorp_overlay', json=omnicorp_input)

    # was the request successful
    assert(response.status_code == 200)

    # load the json
    answer = response.json()

    response = client.post('/weight_correctness', json=answer)

    # was the request successful
    assert(response.status_code == 200)

    # load the json
    answer = response.json()

    # make a good request
    response = client.post('/score', json=answer)

    # was the request successful
    assert(response.status_code == 200)

    # convert the response to a json object
    j_ret = json.loads(response.content)

    # check the data
    ret = j_ret['message']

    # make sure we got back the query_graph, knowledge_graph and results data
    assert(len(ret) == 3)

    # make sure we got the expected number of results
    assert(len(ret['results']) == 95)

    # strider should have created knowledge graph nodes and edges
    assert (len(ret['knowledge_graph']['nodes']) > 1)
    assert (len(ret['knowledge_graph']['edges']) > 1)

    # turn dicts into a list for easier indexing
    kg_node_list = list(ret['knowledge_graph']['nodes'].items())
    kg_edge_list = list(ret['knowledge_graph']['edges'].items())

    found = False

    # insure that ranker/omnicorp overlay added the omni article count
    for n in kg_node_list:
        if 'attributes' in n[1] and len(n[1]['attributes']) > 0:
            for a in n[1]['attributes']:
                if a['original_attribute_name'] == 'omnicorp_article_count':
                    found = True
                    break
        if found:
            break

    assert found

    # found = False
    #
    # # insure that ranker/omnicorp overlay added the omnicorp data
    # for e in kg_edge_list:
    #     if 'attributes' in e[1]:
    #         for a in e[1]['attributes']:
    #             if str(a['value']).startswith('omnicorp') or str(a['value']).startswith('omnicorp.term_to_term'):
    #                 found = True
    #                 break
    #     if found:
    #         break
    #
    # assert found

    found = False

    # insure that ranker/weight added the weight element
    for r in ret['results']:
        if 'edge_bindings' in r:
            for nb in r['edge_bindings']:
                if len(r['edge_bindings'][nb][0]['attributes']) > 0 and r['edge_bindings'][nb][0]['attributes'][0]['original_attribute_name'] == 'weight':
                    found = True
                    break
        if found:
            break

    assert found

    found = False

    # insure ranker/score added the score element
    for r in ret['results']:
        if 'score' in r:
            found = True
            break

    assert found
