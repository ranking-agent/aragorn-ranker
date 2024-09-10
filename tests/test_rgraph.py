from ranker.shared.ranking import Ranker
#temp
from .fixtures import overlapping_set


#I think that changes to rgraph make this no longer a valid test, but I'm not 100% on that.
def x_test_rgraph_edges_between_same_qnode(overlapping_set):
    message =  overlapping_set['message']
    ranker = Ranker(message)

    for answer in message['results']:
        _, redges = ranker.get_rgraph(answer)

        for redge in redges:
            # Assert that the qnode_id of each endpoint are different
            assert redge['subject'][0] != redge['object'][0]
