"""ROBOKOP ranking."""

from operator import itemgetter
from collections import defaultdict
from itertools import combinations, permutations, product
import logging
import numpy as np
# import re
# from uuid import uuid4
# from ranker.shared.util import flatten_semilist

logger = logging.getLogger(__name__)


class Ranker:
    """Ranker."""

    def __init__(self, message):
        """Create ranker."""
        self.kgraph = message['knowledge_graph']
        self.qgraph = message['query_graph']

        kedges = self.kgraph['edges']

        attribute = {'name': 'weight', 'value': 1, 'type': 'EDAM:data_0006', 'url': None, 'source': None}

        for kedge in kedges:
            if kedges[kedge]['attributes'] is None:
                kedges[kedge]['attributes'] = [attribute]
            else:
                found = False
                for attrib in kedges[kedge]['attributes']:
                    if attrib['name'] == 'weight':
                        found = True

                if not found:
                    kedges[kedge]['attributes'].append(attribute)

        self.qnode_by_id = {n: self.qgraph['nodes'][n] for n in self.qgraph['nodes']}
        self.qedge_by_id = {n: self.qgraph['edges'][n] for n in self.qgraph['edges']}
        self.kedge_by_id = {n: self.kgraph['edges'][n] for n in kedges}
        self.kedges_by_knodes = defaultdict(list)

        for e in kedges:
            self.kedges_by_knodes[tuple(sorted([kedges[e]['subject'], kedges[e]['object']]))].append(kedges[e])

        # find leaf-set qnodes
        degree = defaultdict(int)
        for edge in self.qgraph['edges']:
            degree[self.qgraph['edges'][edge]['subject']] += 1
            degree[self.qgraph['edges'][edge]['object']] += 1
        self.leaf_sets = [
            node
            for node in self.qgraph['nodes']
            if self.qgraph['nodes'][node].get('is_set', False) and degree[node] == 1
        ]


    def rank(self, answers, jaccard_like=False):
        """Generate a sorted list and scores for a set of subgraphs."""
        # get subgraph statistics
        print(f'{len(answers)} answers')
        answers_ = []
        for answer in answers:
            answers_.append(self.score(answer, jaccard_like=jaccard_like))

        answers.sort(key=itemgetter('score'), reverse=True)
        return answers

    def score(self, answer, jaccard_like=False):
        """Compute answer score."""
        # answer is a list of dicts with fields 'id' and 'bound'
        rgraph = self.get_rgraph(answer)

        laplacian = self.graph_laplacian(rgraph)
        if np.any(np.all(np.abs(laplacian) == 0, axis=0)):
            answer['score'] = 0
            return answer
        nonset_node_ids = [
            idx for idx, rnode_id in enumerate(rgraph[0])
            if (
                (rnode_id[0] not in self.qnode_by_id) or
                (not self.qnode_by_id[rnode_id[0]].get('is_set', False))
            )
        ]

        score = 1 / kirchhoff(laplacian, nonset_node_ids)

        # fail safe to nuke nans
        score = score if np.isfinite(score) and score >= 0 else -1
        if jaccard_like:
            answer['score'] = score / (1 - score)
        else:
            answer['score'] = score
        return answer

    def graph_laplacian(self, rgraph):
        """Generate graph Laplacian."""
        node_ids, edges = rgraph

        # compute graph laplacian for this case with potentially duplicated nodes
        num_nodes = len(node_ids)
        laplacian = np.zeros((num_nodes, num_nodes))
        index = {node_id: node_ids.index(node_id) for node_id in node_ids}
        for edge in edges:
            subject_id, object_id, weight = edge['subject'], edge['object'], edge['weight']
            i, j = index[subject_id], index[object_id]
            laplacian[i, j] += -weight
            laplacian[j, i] += -weight
            laplacian[i, i] += weight
            laplacian[j, j] += weight

        return laplacian

    def get_rgraph(self, answer):
        """Get "ranker" subgraph."""
        rnodes = set()
        redges = []

        # get list of nodes, and knode_map
        knode_map = defaultdict(set)
        for nb in answer['node_bindings']:
            # get the query node binding entry
            qnode_id = nb
            knode_ids: list = []

            # get all the knowledge node entries
            for knode_id in answer['node_bindings'][nb]:
                knode_ids.append(knode_id['id'])

            for knode_id in knode_ids:
                rnode_id = (qnode_id, knode_id)
                rnodes.add(rnode_id)
                knode_map[knode_id].add(rnode_id)

                if qnode_id in self.leaf_sets:
                    anchor_id = (f'{qnode_id}_anchor', '')
                    rnodes.add(anchor_id)
                    redges.append({
                        'weight': 1e9,
                        'subject': rnode_id,
                        'object': anchor_id
                    })
        rnodes = list(rnodes)

        # get "result" edges
        for eb in answer['edge_bindings']:
            qedge_id = eb
            kedges = answer['edge_bindings'][eb]

            # find source and target rnode(s)
            # qedge direction may not match kedge direction
            # we'll go with the kedge direction
            # note that a single support edge may in theory result in multiple redges
            # if the same knode is bound to multiple qnodes

            for kedge_node in kedges:
                kedge = self.kedge_by_id[kedge_node['id']]

                try:
                    qedge = self.qedge_by_id[qedge_id]
                    pairs = list(product(
                        [
                            rnode for rnode in rnodes
                            if rnode[0] in (qedge['subject'], qedge['object']) and rnode[1] == kedge['subject']
                        ],
                        [
                            rnode for rnode in rnodes
                            if rnode[0] in (qedge['subject'], qedge['object']) and rnode[1] == kedge['object']
                        ],
                    ))
                except KeyError:
                    # a support edge
                    # qedge just needs to contain regex patterns for source and target ids
                    pairs = list(product(
                        [
                            rnode for rnode in rnodes
                            if rnode[1] == kedge['subject']
                        ],
                        [
                            rnode for rnode in rnodes
                            if rnode[1] == kedge['object']
                        ],
                    ))

                for subject, object in pairs:
                    edge = {
                        'weight': kedge_node['weight'],
                        'subject': subject,
                        'object': object
                    }
                    redges.append(edge)

        return rnodes, redges


def kirchhoff(L, keep):
    """Compute Kirchhoff index, including only specific nodes."""
    num_nodes = L.shape[0]
    cols = []
    for x, y in combinations(keep, 2):
        d = np.zeros(num_nodes)
        d[x] = -1
        d[y] = 1
        cols.append(d)
    x = np.stack(cols, axis=1)

    return np.trace(x.T @ np.linalg.lstsq(L, x, rcond=None)[0])


def matching_subsets(patterns, superset):
    """Return subsets matching the regular expressions."""
    subsets = []
    for subset in superset:
        if patterns(subset):
            subsets.append(subset)
    return subsets
