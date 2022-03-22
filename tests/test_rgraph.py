from ranker.shared.ranker_obj import Ranker
from .fixtures import overlapping_set

def test_rgraph_edges_between_same_qnode(overlapping_set):
    message =  overlapping_set['message']
    ranker = Ranker(message)

    for answer in message['results']:
        _, redges = ranker.get_rgraph(answer)

        for redge in redges:
            # Assert that the qnode_id of each endpoint are different
            rnodes = redge['rnodes']
            assert min(rnodes) != max(rnodes)

def test_rgraph_rnodes_match_qnodes(overlapping_set):
    message =  overlapping_set['message']
    ranker = Ranker(message)

    for answer in message['results']:
        rnodes, _ = ranker.get_rgraph(answer)

        assert sorted(rnodes) == sorted(answer['node_bindings'].keys())

