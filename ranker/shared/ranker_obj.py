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
        kgraph = message['knowledge_graph']
        qgraph = message['query_graph']

        kedges = kgraph['edges']
        if not any('weight' in kedge for kedge in kedges):
            for kedge in kedges:
                kedge['weight'] = 1

        self.qnode_by_id = {n['id']: n for n in qgraph['nodes']}
        self.kedge_by_id = {n['id']: n for n in kedges}
        self.qedge_by_id = {n['id']: n for n in qgraph['edges']}
        self.kedges_by_knodes = defaultdict(list)
        for e in kedges:
            self.kedges_by_knodes[tuple(sorted([e['source_id'], e['target_id']]))].append(e)

        # find leaf-set qnodes
        degree = defaultdict(int)
        for edge in qgraph['edges']:
            degree[edge['source_id']] += 1
            degree[edge['target_id']] += 1
        self.leaf_sets = [
            node['id']
            for node in qgraph['nodes']
            if node.get('set', False) and degree[node['id']]==1
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
                (not self.qnode_by_id[rnode_id[0]].get('set', False))
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
            source_id, target_id, weight = edge['source_id'], edge['target_id'], edge['weight']
            i, j = index[source_id], index[target_id]
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
            qnode_id = nb['qg_id']
            knode_id = nb['kg_id']
            rnode_id = (qnode_id, knode_id)
            rnodes.add(rnode_id)
            knode_map[knode_id].add(rnode_id)
            if qnode_id in self.leaf_sets:
                anchor_id = (f'{qnode_id}_anchor', '')
                rnodes.add(anchor_id)
                redges.append({
                    'weight': 1e9,
                    'source_id': rnode_id,
                    'target_id': anchor_id
                })
        rnodes = list(rnodes)

        # get "result" edges
        for eb in answer['edge_bindings']:
            kedge_id = eb['kg_id']
            qedge_id = eb['qg_id']

            # find source and target rnode(s)
            # qedge direction may not match kedge direction
            # we'll go with the kedge direction
            # note that a single support edge may in theory result in multiple redges
            # if the same knode is bound to multiple qnodes

            kedge = self.kedge_by_id[kedge_id]
            try:
                qedge = self.qedge_by_id[qedge_id]
                pairs = list(product(
                    [
                        rnode for rnode in rnodes
                        if rnode[0] in (qedge['source_id'], qedge['target_id']) and rnode[1] == kedge['source_id']
                    ],
                    [
                        rnode for rnode in rnodes
                        if rnode[0] in (qedge['source_id'], qedge['target_id']) and rnode[1] == kedge['target_id']
                    ],
                ))
            except KeyError:
                # a support edge
                # qedge just needs to contain regex patterns for source and target ids
                pairs = list(product(
                    [
                        rnode for rnode in rnodes
                        if rnode[1] == kedge['source_id']
                    ],
                    [
                        rnode for rnode in rnodes
                        if rnode[1] == kedge['target_id']
                    ],
                ))

            for source_id, target_id in pairs:
                edge = {
                    'weight': eb['weight'],
                    'source_id': source_id,
                    'target_id': target_id
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
