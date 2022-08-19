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
                    if attrib['original_attribute_name'] == 'weight':
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

        rnodes, redges = self.get_rgraph(answer)

        laplacian = self.graph_laplacian((rnodes, redges))
        if np.any(np.all(np.abs(laplacian) == 0, axis=0)):
            answer['score'] = 0
            return answer

        probe_positions = range(len(rnodes))
        score = 1 / kirchhoff(laplacian, probe_positions)

        # fail safe to nuke nans
        score = score if np.isfinite(score) and score >= 0 else -1

        if jaccard_like:
            answer['score'] = score / (1 - score)
        else:
            answer['score'] = score
        return answer

    def graph_laplacian(self, rgraph):
        """Generate graph Laplacian."""
        rnodes, redges = rgraph
        rnode_indices = {node: idx for idx, node in enumerate(rnodes)}
        num_nodes = len(rnodes)
            
        laplacian = np.zeros((num_nodes, num_nodes))
        for redge in redges:
            i, j = [rnode_indices[rnode] for rnode in redge['rnodes']]
            weight = redge['weight']
            laplacian[i, j] += -weight
            laplacian[j, i] += -weight
            laplacian[i, i] += weight
            laplacian[j, j] += weight

        return laplacian

    def merge_weights(self, weights, across=None):
        if across == 'knode_pairs':
            return np.mean(weights)
        elif across == 'sources':
            return sum(weights)
        elif across == 'attributes':
            return max(weights)
        else:
            raise ValueError(across)

    def get_rgraph(self, answer):
        """
        Builds the ranker graph.

        The ranker graph (rgraph) is a weighted version of the question graph. 
        To compute the weights of each qedge, we aggregate weights from multiple
        sources. There are several one-to-many relationships between the given
        weights and the qedges.

          1. A single qedge can be bound to multiple kedges.

          2. A single kedge can have weights from multiple sources.

          3. Weights of a single source can come from multiple attributes.

        We repeatedly merge the weights from the attribute level up to the qedge
        level using `merge_weights` to assign each qedge a single weight.
        """

        # Create ranker graph nodes; one node per qnode_id
        rnodes = list(answer['node_bindings'].keys())

        # Build mapping to and from qnodes and their knodes
        knode_to_qnodes = defaultdict(lambda: [])
        qnode_to_knodes = defaultdict(lambda: set())
        for qnode_id, qnode_bindings in answer['node_bindings'].items():
            for qnode_binding in qnode_bindings:
                knode_to_qnodes[qnode_binding['id']].append(qnode_id)
                qnode_to_knodes[qnode_id].add(qnode_binding['id'])

        # Tree of pre-merge edge weights where for each edge/qnode pair, we branch on
        # (1) distinct knode pairs for each qnode pair
        # (2) distinct weight sources for each knode pair
        # which then map to weights (from different attributes) for each weight source
        weight_tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        # Loop over each kedge bound to this result
        for qedge_id, kedge_bindings in answer['edge_bindings'].items():
            for kedge_binding in kedge_bindings:
                kedge = self.kedge_by_id[kedge_binding['id']]
                ksubject, kobject = kedge['subject'], kedge['object']

                # Skip this kedge if it has no attributes (meaning no weight)
                if kedge_binding.get('attributes') is None:
                    continue

                # Get the sources and weights of this kedge 
                weights = defaultdict(lambda: [])
                for attribute in kedge_binding['attributes']:
                    # Search for the weight attribute
                    if attribute['original_attribute_name'].startswith('weight'):
                        source = 'unspecified'
                        if attribute.get('attributes', []) is not None:
                            for sub_attr in attribute.get('attributes', []):
                                if sub_attr.get('original_attribute_name', None) == 'aragorn_weight_source':
                                    source = sub_attr.get('value', source)
                                    break

                        weights[source].append(attribute['value'])

                # Skip this kedge if it has no weights
                if len(weights) == 0:
                    continue


                if qedge_id in self.qedge_by_id:
                    # If the kedge is bound to a qedge, just use the endpoints of the qedge
                    qedge = self.qedge_by_id[qedge_id]
                    qsubject, qobject = qedge['subject'], qedge['object']
                    """
                    This kedge can create up to 2 edges in our pre-merged rgraph

                    Consider this question graph: n0 --e0--> n1.

                    Suppose n0 is bound to D1 and D2, n1 is bound to D1 and D2,
                    and e0 is bound to this knowledge graph edge: D1 --> D2.

                    Following the direction of the kedge, we expect the
                    following edges in the rgraph:
                    (n0, D1) --> (n1, D2)
                    (n1, D1) --> (n0, D2)
                    """
                    knode_pairs = []
                    if ksubject in qnode_to_knodes[qsubject] and kobject in qnode_to_knodes[qobject]:
                        knode_pairs.append(frozenset(((qsubject, ksubject), (qobject, kobject))))
                    if ksubject in qnode_to_knodes[qobject] and kobject in qnode_to_knodes[qsubject]:
                        knode_pairs.append(frozenset(((qobject, ksubject), (qsubject, kobject))))
                else:
                    # Support edges aren't bound to particular qnodes, so we place them everywhere they fit
                    """
                    We store both the qnode_id and knode_id in each knode_pair
                    because a pair of knode_id is insufficient. Consider this
                    example:
                    (n0, D1) --> (n1, D2)
                    (n1, D1) --> (n0, D2)

                    These are different edges, but storing only knode_id info
                    would treat them as identical.
                    """
                    knode_pairs = set(frozenset(pair) for pair in product(
                        [(qnode_id, ksubject) for qnode_id in knode_to_qnodes[ksubject]],
                        [(qnode_id, kobject) for qnode_id in knode_to_qnodes[kobject]]
                    ) if pair[0][0] != pair[1][0])  # Excludes edges between qnode_id/knode_id pairs with same qnode_id (e.g., n1 -> n1)
                    

                for knode_pair in knode_pairs:
                    qnode_pair = frozenset((min(knode_pair)[0], max(knode_pair)[0]))
                    for source, values in weights.items():
                        weight_tree[qnode_pair][knode_pair][source].extend(values)

        # Now let's merge weights of the tree from bottom to top
        redges = []
        for qnode_pair in weight_tree:
            # Merge weights with the same qnodes but different knodes
            knode_pair_weights = []
            for knode_pair in weight_tree[qnode_pair]:
                # Merge weights with the same qnodes and knodes from different sources
                source_weights = []
                for source, weights in weight_tree[qnode_pair][knode_pair].items():
                    # Merge weights with the same qnodes, knodes, and source from different attributes
                    source_weights.append(self.merge_weights(weights, 'attributes'))
                knode_pair_weights.append(self.merge_weights(source_weights, 'sources'))
            redges.append({
                'rnodes': qnode_pair,
                'weight': self.merge_weights(knode_pair_weights, 'knode_pairs')
            })

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
