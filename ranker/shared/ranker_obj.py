"""ROBOKOP ranking."""

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
        self.kgraph = message.get("knowledge_graph", {"nodes": {}, "edges": {}})
        self.qgraph = message.get("query_graph", {"nodes":{}, "edges":{}} )

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

        return answers

    def score(self, answer, jaccard_like=False):
        """Compute answer score."""
        # answer is a list of dicts with fields 'id' and 'bound'
        r_node_ids, edges_all = self.get_rgraph(answer)
        
        

        
        for i_analysis,edges in enumerate(edges_all):
            laplacian = self.graph_laplacian((r_node_ids,edges), answer)
            if np.any(np.all(np.abs(laplacian) == 0, axis=0)):
                answer['analyses'][i_analysis]['score'] = 0
                continue
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
            probes = list(combinations(rgraph_probe_nodes,2))
            score = np.exp(-kirchhoff(laplacian, probes))

            # fail safe to nuke nans
            score = score if np.isfinite(score) and score >= 0 else -1

            if jaccard_like:
                answer['analyses'][i_analysis]['score'] = score / (1 - score)
            else:
                answer['analyses'][i_analysis]['score'] = score
        return answer

    def graph_laplacian(self, rgraph, answer):
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
        laplacian = np.zeros((num_nodes, num_nodes))
        for subject_index in range(num_nodes):
            q_node_id_subject = node_ids[subject_index][0]
            for object_id in range(num_nodes):
                q_node_id_object = node_ids[object_id][0]
                edge_qnode_ids = frozenset((q_node_id_subject, q_node_id_object))

                # Set default weight (or 0 when edge is not a qedge)
                weight = self.DEFAULT_WEIGHT if edge_qnode_ids in qedge_qnode_ids else 0.0

                for source in weight_dict[subject_index][object_id].keys():
                    for property in weight_dict[subject_index][object_id][source].keys():
                        source_w = weight_dict[subject_index][object_id][source][property]
                        source_weighted = source_w * source_weight(source, property, source_weights=self.source_weights, unknown_source_weight=self.unknown_source_weight)
                        if source_weighted==1:
                            source_weighted=0.99999999
                        weight = weight + -1/(np.log(source_weighted))
                
                laplacian[subject_index, object_id] += -weight
                laplacian[object_id, subject_index] += -weight
                laplacian[subject_index, subject_index] += weight
                laplacian[object_id, object_id] += weight
        
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
                        'weight': {"anchor_node":{"anchor_node":1e9}},
                        'subject': rnode_id,
                        'object': anchor_id
                    })
        rnodes = list(rnodes)

        # for eb in answer['edge_bindings']:
        # qedge_id = eb
        # kedges = answer['edge_bindings'][eb]
        analysis_edges = []
        for i_analysis in range(len(answer.get('analyses',[]))):
        # get "result" edges
            for qedge_id, kedge_bindings in answer['analyses'][i_analysis]['edge_bindings'].items():
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
            analysis_edges.append(redges)

        return rnodes, analysis_edges


def kirchhoff(L, probes):
    """Compute Kirchhoff index, including only specific nodes."""
    try:
        num_nodes = L.shape[0]
        cols = []
        for x, y in probes:
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
                if p.get("original_attribute_name","") == "omnicorp_article_count":
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
        
        #Get source information
        edge_info_final = "unspecified"
        sources = edges[edge].get("sources", None)
        if sources is not None:
            for source in sources:
                if 'primary_knowledge_source' in source.get("resource_role",None):
                    edge_info_final = source.get("resource_id","unspecified")


            
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
                        if type(attribute["value"], list):
                            p_value = attribute["value"][0] if len(attribute["value"]) > 0 else None
                        else:
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
                    if type(attribute["value"], list):
                        p_value = attribute["value"][0] if len(attribute["value"]) > 0 else None
                    else:
                        p_value = attribute["value"]
                
            edge_info = {}
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
