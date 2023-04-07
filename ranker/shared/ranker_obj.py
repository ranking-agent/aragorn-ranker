"""ROBOKOP ranking."""

import itertools
import logging
from collections import defaultdict
from itertools import combinations, product
from operator import itemgetter

import numpy as np
from ranker.shared.sources import source_weight, get_profile, source_sigmoid, BLENDED_PROFILE, CLINICAL_PROFILE, CORRELATED_PROFILE, CURATED_PROFILE

import re

logger = logging.getLogger(__name__)

class Ranker:
    """Ranker."""

    DEFAULT_WEIGHT = 1e-2

    def __init__(self, message, profile = "blended"):
        """Create ranker."""
        self.kgraph = message['knowledge_graph']
        self.qgraph = message['query_graph']

        source_weights, unknown_source_weight, source_transformation, unknown_source_transformation = get_profile(profile)
        self.source_weights = source_weights
        self.unknown_source_weight = unknown_source_weight
        self.source_transformation = source_transformation
        self.unknown_source_transformation = unknown_source_transformation
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
        self.node_pubs = get_node_pubs(self.kgraph)
        self.rank_vals = get_vals(kedges, self.node_pubs, self.source_transformation, self.unknown_source_transformation)
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

        weighted_graph = self.weight_graph(rgraph, answer)
        # if np.any(np.all(np.abs(weighted_graph) == 0, axis=0)):
        #     answer['score'] = 0
        #     return answer
        r_node_ids, edges = rgraph
        index = {node_id[0]: r_node_ids.index(node_id) for node_id in r_node_ids}
        q_node_ids = list(self.qgraph['nodes'].keys())
        n_q_nodes = len(q_node_ids)
        q_conn = np.full((n_q_nodes, n_q_nodes),0)
        for e in self.qgraph['edges'].values():
            e_sub = q_node_ids.index(e['subject'])
            e_obj = q_node_ids.index(e['object'])
            if e_sub is not None and e_obj is not None:
                q_conn[e_sub, e_obj] = 1

        node_conn = np.sum(q_conn,0) + np.sum(q_conn,1).T
        probe_nodes = []
        for conn in range(np.max(node_conn)):
            is_this_conn = node_conn == (conn+1)
            probe_nodes += list(np.where(is_this_conn)[0])
            if len(probe_nodes) > 1:
                break
        #converting qgraph inds to rgraph inds:
        
        rgraph_inds = []
        for node_ind in range(len(q_node_ids)):
            node_label = q_node_ids[node_ind]
            rgraph_inds.append(index[node_label])
        
        rgraph_probe_nodes = [rgraph_inds[i] for i in probe_nodes]
        probes = list(itertools.combinations(rgraph_probe_nodes,2))
        
        measurement = [ranker.path_collapse(weighted_graph, probe) for probe in probes]

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
            score = np.mean(measurement)

        # fail safe to nuke nans
        score = score if np.isfinite(score) and score >= 0 else -1

        if jaccard_like:
            answer['score'] = score / (1 - score)
        else:
            answer['score'] = score
        return answer

    def path_collapse(self, weighted_graph, probe):
        if probe[0] == probe[1]:
            return 1
        if probe[0]>probe[1]:
            probe = (probe[1],probe[0])
        parallel_steps = [(i,w) for i, w in enumerate(weighted_graph[probe[0],:]) if w > 0]
        parallel_parts = []
        if parallel_steps:
            for step in parallel_steps:
                new_p = (step[0], probe[1])
                parallel_parts.append(self.series_combine([step[1], self.path_collapse(weighted_graph, new_p)]))
                
            out = self.parallel_combine(parallel_parts)
            return out
    
    def series_combine(self, ws):
        return np.prod(ws)

    def parallel_combine(self, ws):
        out = 0
        for i, w in enumerate(ws):
            remaining = 1
            for w2 in ws[:i]:
                remaining = remaining * (1 - w2)
            out = out + remaining*w
        return out

    def weight_graph(self, rgraph, answer):
        """Generate graph Laplacian."""
        node_ids, edges = rgraph

        # compute graph laplacian for this case while removing duplicate sources for each edge in the result graph
        num_nodes = len(node_ids)
        weight_dict = []
        for subject_index in range(num_nodes):
            weight_dict_i = []
            for object_id in range(num_nodes):
                weight_dict_i.append({})
            weight_dict.append(weight_dict_i)

        index = {node_id: node_ids.index(node_id) for node_id in node_ids}
        for edge in edges:
            subject_id, object_id, edge_weight = edge['subject'], edge['object'], edge['weight']
            subject_index, object_id = index[subject_id], index[object_id]
            for edge_source in edge_weight.keys():
                for edge_property in edge_weight[edge_source].keys():
                    val = edge_weight[edge_source][edge_property]
                    if edge_source in weight_dict[subject_index][object_id]:
                        if edge_property in weight_dict[subject_index][object_id][edge_source]:
                            weight_dict[subject_index][object_id][edge_source][edge_property] = max(weight_dict[subject_index][object_id][edge_source][edge_property], val)
                        else:
                            weight_dict[subject_index][object_id][edge_source][edge_property] = val
                    else:
                        weight_dict[subject_index][object_id][edge_source] = {}
                        weight_dict[subject_index][object_id][edge_source][edge_property] = val

        qedge_qnode_ids = set([frozenset((e['subject'], e['object'])) for e in self.qedge_by_id.values()])
        weighted_graph = np.zeros((num_nodes, num_nodes))
        for subject_index in range(num_nodes):
            q_node_id_subject = node_ids[subject_index][0]
            for object_id in range(num_nodes):
                q_node_id_object = node_ids[object_id][0]
                edge_qnode_ids = frozenset((q_node_id_subject, q_node_id_object))

                # Set default weight (or 0 when edge is not a qedge)
                weight = self.DEFAULT_WEIGHT if edge_qnode_ids in qedge_qnode_ids else 0

                for source in weight_dict[subject_index][object_id].keys():
                    for property in weight_dict[subject_index][object_id][source].keys():
                        source_w = weight_dict[subject_index][object_id][source][property]
                        weight = weight + (1-weight)*source_w * source_weight(source, property, source_weights=self.source_weights, unknown_source_weight=self.unknown_source_weight)
                        # I think we don't actually want to take the max here 
                        # weight = max(weight, source_w * source_weight(source, property, source_weights=self.source_weights, unknown_source_weight=self.unknown_source_weight))
                # This puts it in the weighted graph and ensures upper triangular weighted_graph
                if subject_index<object_id:
                    weighted_graph[subject_index, object_id] = weighted_graph[subject_index, object_id] + (1-weighted_graph[subject_index, object_id])*weight                     
                else:
                    weighted_graph[object_id, subject_index] = weighted_graph[object_id, subject_index] + (1-weighted_graph[object_id, subject_index])*weight
                    
        
        return weighted_graph

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
                        'weight': {"anchor_node":{"anchor_node":1e9}},
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
                    edge_vals = self.rank_vals.get(kedge_binding['id'],None)
                    if edge_vals is not None:
                        edge = {
                            'weight': {edge_vals['source']: {k:v for k,v in edge_vals.items() if k != 'source'}},
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

def get_node_pubs(kgraph):
    node_pubs: dict = {}
    for n in kgraph["nodes"]:
            # init the count value
            omnicorp_article_count: int = 0
            if not kgraph["nodes"][n].get("attributes"):
                kgraph["nodes"][n]["attributes"] = []
            # get the article count atribute
            for p in kgraph["nodes"][n]["attributes"]:
                # is this what we are looking for
                if p["original_attribute_name"] == "omnicorp_article_count":
                    # save it
                    omnicorp_article_count = p["value"]

                    # no need to continue
                    break

            # add the node d and count to the dict
            node_pubs.update({n: omnicorp_article_count})
    return node_pubs

def get_vals(edges, node_pubs,source_transfroamtion, unknown_source_transformation):
    # constant count of all publications
    all_pubs = 27840000
    
    # get the knowledge graph edges
    #edges = kgraph["edges"]
    edge_vals = {}
    # for each knowledge graph edge
    for edge in edges:
        # We are getting some results back (BTE?) that have "publications": ['PMID:1234|2345|83984']
        attributes = edges[edge].get("attributes", None)

        # init storage for the publications and their count
        publications = []
        num_publications = 0
            
        p_value = None    
        if attributes is not None:
            # for each data attribute collect the needed params
            for attribute in attributes:
                # This picks up omnicorp
                if attribute.get("original_attribute_name", None) is not None:
                    # is this the publication list
                    if attribute["original_attribute_name"].startswith(
                        "publications"
                    ):
                        publications = attribute["value"]
                    # else is this the number of publications
                    elif attribute["original_attribute_name"].startswith(
                        "num_publications"
                    ):
                        num_publications = attribute.get("value", 0)
                    elif 'p_value' in attribute["original_attribute_name"] or 'p-value' in attribute["original_attribute_name"]:
                        p_value = attribute["value"]
                # This picks up Text Miner KP
                elif (
                    attribute["attribute_type_id"] == "biolink:supporting_document"
                ):
                    publications = attribute["value"]
                    if isinstance(publications, str):
                        publications = [publications]
                # This picks up how BTE returns pubs
                elif attribute["attribute_type_id"] == "biolink:publications":
                    publications = attribute["value"]
                elif 'p_value' in attribute["attribute_type_id"] or 'p-value' in attribute["attribute_type_id"]:
                    p_value = attribute["value"]
                

            # Record the source of origination
            edge_info = {
                "biolink:aggregator_knowledge_source": "not_found",
                "biolink:original_knowledge_source": "not_found",
                "biolink:primary_knowledge_source": "not_found",
            }
            for attribute in reversed(attributes):
                if attribute.get("attribute_type_id", None) is not None:
                    if attribute["attribute_type_id"] in edge_info.keys():
                        v = attribute.get("value", None)
                        if type(v) is list:
                            v = v[0]
                        if v is not None:
                            edge_info[attribute["attribute_type_id"]] = v
                        else:
                            edge_info[
                                attribute["attribute_type_id"]
                            ] = "unspecified"

            if edge_info["biolink:original_knowledge_source"] != "not_found":
                edge_info_final = edge_info["biolink:original_knowledge_source"]
            elif edge_info["biolink:primary_knowledge_source"] != "not_found":
                edge_info_final = edge_info["biolink:primary_knowledge_source"]
            elif edge_info["biolink:aggregator_knowledge_source"] != "not_found":
                edge_info_final = edge_info["biolink:aggregator_knowledge_source"]
            else:
                edge_info_final = "unspecified"

            # if there was only 1 publication value found insure it wasnt a character separated list
            if len(publications) == 1:
                if "|" in publications[0]:
                    publications = publications[0].split("|")
                elif "," in publications[0]:
                    publications = publications[0].split(",")

                # get the real publication count
                num_publications = len(publications)

            # if there was no publication count found revert to the number of individual values
            if num_publications == 0:
                num_publications = len(publications)
            literature_coocurrence = None
            if (
                    edges[edge].get("predicate")
                    == "biolink:occurs_together_in_literature_with"
                ):
                    subject_pubs = int(node_pubs[edges[edge]["subject"]])
                    object_pubs = int(node_pubs[edges[edge]["object"]])

                    cov = (num_publications / all_pubs) - (subject_pubs / all_pubs) * (
                        object_pubs / all_pubs
                    )
                    cov = max((cov, 0.0))
                    literature_coocurrence = cov * all_pubs
                    effective_pubs = None
            else:
                effective_pubs = num_publications + 1  # consider the curation a pub
            edge_vals[edge] = {}
            if p_value is not None:
                edge_vals[edge]['p-value'] = source_sigmoid(p_value, edge_info_final, "p-value", source_transformation=source_transfroamtion, unknown_source_transformation=unknown_source_transformation)
            if literature_coocurrence is not None:
                edge_vals[edge]['literature_co-occurrence'] = source_sigmoid(literature_coocurrence, edge_info_final, "literature_co-occurrence", source_transformation=source_transfroamtion, unknown_source_transformation=unknown_source_transformation)
            if effective_pubs is not None:
                edge_vals[edge]['publications'] = source_sigmoid(effective_pubs, edge_info_final, "publications", source_transformation=source_transfroamtion, unknown_source_transformation=unknown_source_transformation)
            edge_vals[edge]['source'] = edge_info_final
    return edge_vals
