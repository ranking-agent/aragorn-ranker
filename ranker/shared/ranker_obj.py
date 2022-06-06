"""ROBOKOP ranking."""

import logging
from collections import defaultdict
from itertools import combinations, permutations, product
from operator import itemgetter

import numpy as np
from ranker.shared.sources import source_weight

logger = logging.getLogger(__name__)

class Ranker:
    """Ranker."""

    DEFAULT_WEIGHT = 1e-2

    def __init__(self, message):
        """Create ranker."""
        self.kgraph = message['knowledge_graph']
        self.qgraph = message['query_graph']

        kedges = self.kgraph['edges']

        attribute = {'original_attribute_name': 'weight',
                     'attribute_type_id': 'biolink:has_numeric_value',
                     'value': 1,
                     'value_type_id': 'EDAM:data_1669',
                     'value_url': None,
                     'attribute_source': None}

        for kedge in kedges:
            if kedges[kedge].get('attributes') is None:
                kedges[kedge]['attributes'] = [attribute]
            else:
                found = False
                for attrib in kedges[kedge]['attributes']:
                    if attrib.get('original_attribute_name', None) == 'weight':
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
        #print(f'{len(answers)} answers')
        answers_ = []
        for answer in answers:
            answers_.append(self.score(answer, jaccard_like=jaccard_like))

        answers.sort(key=itemgetter('score'), reverse=True)
        return answers

    def score(self, answer, jaccard_like=False):
        """Compute answer score."""
        # answer is a list of dicts with fields 'id' and 'bound'
        rgraph = self.get_rgraph(answer)

        laplacian = self.graph_laplacian(rgraph, answer)
        if np.any(np.all(np.abs(laplacian) == 0, axis=0)):
            answer['score'] = 0
            return answer

        #We want nodes that are not sets.  We could look in the QG, but that doesn't work very well because if only a single node is bound
        # in the answer, we want to consider that a non-set, even if the qg node is a set.  So let's just look at how many are bound.
        # rgraph[0] is structured as a list of (qg_id, kg_id) tuples.  So we want the indices of rgraph[0] that have qg_id that occur only once in rgraph[0]
        counts = defaultdict(int)
        for q,k in rgraph[0]:
            counts[q] += 1
        nonset_node_ids = [ idx for idx,rnode_id in enumerate(rgraph[0]) if ( counts[rnode_id[0]] == 1 ) ]
        #nonset_node_ids = [
        #    idx for idx, rnode_id in enumerate(rgraph[0])
        #    if (
        #        (rnode_id[0] not in self.qnode_by_id) or
        #        (not self.qnode_by_id[rnode_id[0]].get('is_set', False))
        #    )
        #]

        # the nonset node count has to be at least 2 nodes
        if len(nonset_node_ids) < 2:
            score = 0
        else:
            score = 1 / kirchhoff(laplacian, nonset_node_ids)

        # fail safe to nuke nans
        score = score if np.isfinite(score) and score >= 0 else -1

        if jaccard_like:
            answer['score'] = score / (1 - score)
        else:
            answer['score'] = score
        return answer

    def graph_laplacian(self, rgraph, answer):
        """Generate graph Laplacian."""
        node_ids, edges = rgraph

        # compute graph laplacian for this case while removing duplicate sources for each edge in the result graph
        num_nodes = len(node_ids)
        weight_dict = []
        for i in range(num_nodes):
            weight_dict_i = []
            for j in range(num_nodes):
                weight_dict_i.append({})
            weight_dict.append(weight_dict_i)

        index = {node_id: node_ids.index(node_id) for node_id in node_ids}
        for edge in edges:
            subject_id, object_id, edge_weight = edge['subject'], edge['object'], edge['weight']
            i, j = index[subject_id], index[object_id]
            for k, v in edge_weight.items():
                if k in weight_dict[i][j]:
                    weight_dict[i][j][k] = max(weight_dict[i][j][k], v)
                else:
                    weight_dict[i][j][k] = v

        qedge_qnode_ids = set([frozenset((e['subject'], e['object'])) for e in self.qedge_by_id.values()])
        laplacian = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            q_node_id_i = node_ids[i][0]
            for j in range(num_nodes):
                q_node_id_j = node_ids[j][0]
                edge_qnode_ids = frozenset((q_node_id_i, q_node_id_j))

                # Set default weight (or 0 when edge is not a qedge)
                weight = self.DEFAULT_WEIGHT if edge_qnode_ids in qedge_qnode_ids else 0

                for source, source_w in weight_dict[i][j].items():
                    weight = weight + source_w * source_weight(source)
                laplacian[i, j] += -weight
                laplacian[j, i] += -weight
                laplacian[i, i] += weight
                laplacian[j, j] += weight

        return laplacian

    def get_rgraph(self, answer):
        """Get "ranker" subgraph."""
        rnodes = set()
        redges = []

        # Checks if multiple nodes share node bindings 
        rgraph_sets = [
            node
            for node in answer['node_bindings']
            if len(answer['node_bindings'][node]) > 1 
        ]
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

                if qnode_id in rgraph_sets:
                    anchor_id = (f'{qnode_id}_anchor', '')
                    rnodes.add(anchor_id)
                    redges.append({
                        'weight': {"anchor_node": 1e9},
                        'subject': rnode_id,
                        'object': anchor_id
                    })
        rnodes = list(rnodes)

        # for eb in answer['edge_bindings']:
        # qedge_id = eb
        # kedges = answer['edge_bindings'][eb]

        # get "result" edges
        for qedge_id, kedge_bindings in answer['edge_bindings'].items():
            for kedge_binding in kedge_bindings:
                """
                The code for generating pairs below appears unusual, but we need
                it in order to properly handle unlikely set situations.

                Consider this question graph: n0 --e0--> n1.

                Suppose n0 is bound to D1 and D2, n1 is bound to D1 and D2, and
                e0 is bound to this knowledge graph edge: D1 --> D2.

                Following the direction of the kedge, we expect the following
                edges in the rgraph:
                (n0, D1) --> (n1, D2)
                (n1, D1) --> (n0, D2)

                The `product` used below can generate these pairs, but it can
                also create the following edges:
                (n1, D1) --> (n1, D2)
                (n0, D1) --> (n0, D2)
            
                The `if pair[0][0] != pair[1][0]` prevents this from happening.
                """
                kedge = self.kedge_by_id[kedge_binding['id']]
                ksubject, kobject = kedge['subject'], kedge['object']
                try:
                    qedge = self.qedge_by_id[qedge_id]
                    qedge_nodes = qedge['subject'], qedge['object']
                    pairs = [pair for pair in product(
                        [
                            rnode for rnode in rnodes
                            if rnode[0] in qedge_nodes and rnode[1] == ksubject
                        ],
                        [
                            rnode for rnode in rnodes
                            if rnode[0] in qedge_nodes and rnode[1] == kobject
                        ],
                    ) if pair[0][0] != pair[1][0]]
                except KeyError:
                    # Support edges aren't bound to particular qnode_ids, so let's find all the places they can go
                    # set(tuple(sorted(pair)) for ...) prevents duplicate edges in opposite direction when kedge['subject'] == kedge['object']
                    pairs = set(tuple(sorted(pair)) for pair in product(
                        [
                            rnode for rnode in rnodes
                            if rnode[1] == ksubject
                        ],
                        [
                            rnode for rnode in rnodes
                            if rnode[1] == kobject
                        ],
                    ) if pair[0][0] != pair[1][0]) # Prevents edges between nodes of the same qnode_id

                for subject, object in pairs:
                    # get the weight from the edge binding
                    if kedge_binding.get('attributes') is not None:
                        for item in kedge_binding['attributes']:
                            # search for the weight attribute
                            if item['original_attribute_name'].startswith('weight'):
                                
                                source_key = 'unspecified'
                                if item.get('attributes',[]) is not None:
                                    for sub_attr in item.get('attributes',[]):
                                        if sub_attr.get('original_attribute_name',None) == 'aragorn_weight_source':
                                            source_key = sub_attr.get('value',source_key)
                                            break

                                edge = {
                                    'weight': {source_key: item['value']},
                                    'subject': subject,
                                    'object': object
                                }

                                redges.append(edge)

        return rnodes, redges


def kirchhoff(L, keep):
    """Compute Kirchhoff index, including only specific nodes."""
    try:
        num_nodes = L.shape[0]
        cols = []
        for x, y in combinations(keep, 2):
            d = np.zeros(num_nodes)
            d[x] = -1
            d[y] = 1
            cols.append(d)
        x = np.stack(cols, axis=1)
    except:
        print(cols)
        return

    return np.trace(x.T @ np.linalg.lstsq(L, x, rcond=None)[0])


def matching_subsets(patterns, superset):
    """Return subsets matching the regular expressions."""
    subsets = []
    for subset in superset:
        if patterns(subset):
            subsets.append(subset)
    return subsets
