import json
from fastapi.testclient import TestClient
from ranker.server import APP
#temp
#from .fixtures import svc_test, acet

client = TestClient(APP)

def xtest_service(svc_test):
    """cascade through all ranker end points."""
    response = client.post('/omnicorp_overlay', json=svc_test)

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

#depends on live omnicorp overlay so not turned on by default.  But is a good test
def xtest_omnicorp_overlay_publications(acet):
    """Test that omnicorp_overlay() runs without errors."""
    response = client.post('/omnicorp_overlay', json=acet)

    # load the json
    answer = response.json()

    # make sure the there are the same number of results that went in
    assert(len(answer['message']['results']) == len(acet['message']['results']))

    # assert there are node bindings
    assert(len(answer['message']['results'][0]['node_bindings']) == 2)

    # assert there are 2 edge bindings.  The original one, and the one that came out of omnicorp
    assert(len(answer['message']['results'][0]['edge_bindings']) == 2)

    #Each node in the kgraph should have an omnicorp article count
    for nodeid,node in answer['message']['knowledge_graph']['nodes'].items():
        found = False
        for att in node.get('attributes', {}):
            if att['original_attribute_name'] == 'omnicorp_article_count':
                assert att.get('value',-1) > 0
                found = True
        assert found

    #each new edge should have the right predicate and counts.
    support_edges = set()
    for edgeid, edge in answer['message']['knowledge_graph']['edges'].items():
        if edgeid in acet['message']['knowledge_graph']['edges']:
            #this is an old edge
            continue
        #this is a new edge
        support_edges.add(edgeid)
        #check predicate
        assert edge['predicate'] == 'biolink:occurs_together_in_literature_with'
        #check attribute for shared count
        found = False
        for att in edge.get('attributes', {}):
            if att.get('original_attribute_name','') == 'num_publications':
                assert att.get('value',-1) > 0
                found = True
        assert found

    #Now send the overlaid thing to get weighted
    response = client.post('/weight_correctness', json=answer)
    assert(response.status_code == 200)
    weighted = response.json()

    #We want to make sure that weighting accomplished something.  If things are borked in weighting either we won't get
    # back a weight, or it will be set to a default value of 1.
    for result in weighted['message']['results']:
        for eb_id, edges in result['edge_bindings'].items():
            for edge in edges:
                if edge['id'] in support_edges:
                    weight = 1
                    for att in edge.get('attributes',[]):
                        if att.get('original_attribute_name','') == 'weight':
                            weight = att.get('value',1)
                    assert weight != 1