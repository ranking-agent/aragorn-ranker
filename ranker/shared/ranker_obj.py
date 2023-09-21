"""ROBOKOP ranking."""

import copy
import logging
from collections import defaultdict
from itertools import combinations, product

import numpy as np

from ranker.shared.sources import get_profile, source_sigmoid, source_weight

logger = logging.getLogger(__name__)


class Ranker:
    """Ranker."""

    DEFAULT_WEIGHT = 1e-2

    def __init__(self, message, profile="blended"):
        """Create ranker."""
        self.kgraph = message.get("knowledge_graph", {"nodes": {}, "edges": {}})
        self.qgraph = message.get("query_graph", {"nodes": {}, "edges": {}})
        self.agraphs = message.get("auxiliary_graphs", {})
        (
            source_weights,
            unknown_source_weight,
            source_transformation,
            unknown_source_transformation,
        ) = get_profile(profile)
        self.source_weights = source_weights
        self.unknown_source_weight = unknown_source_weight
        self.source_transformation = source_transformation
        self.unknown_source_transformation = unknown_source_transformation
        kedges = self.kgraph["edges"]

        attribute = {
            "original_attribute_name": "weight",
            "attribute_type_id": "biolink:has_numeric_value",
            "value": 1,
            "value_type_id": "EDAM:data_1669",
            "value_url": None,
            "attribute_source": None,
        }

        for kedge in kedges:
            if kedges[kedge].get("attributes") is None:
                kedges[kedge]["attributes"] = [attribute]
            else:
                found = False
                for attrib in kedges[kedge]["attributes"]:
                    if attrib.get("original_attribute_name", None) == "weight":
                        found = True

                if not found:
                    kedges[kedge]["attributes"].append(attribute)
        self.node_pubs = get_node_pubs(self.kgraph)
        self.rank_vals = get_vals(
            kedges,
            self.node_pubs,
            self.source_transformation,
            self.unknown_source_transformation,
        )
        self.qnode_by_id = {n: self.qgraph["nodes"][n] for n in self.qgraph["nodes"]}
        self.qedge_by_id = {n: self.qgraph["edges"][n] for n in self.qgraph["edges"]}
        self.kedge_by_id = {n: self.kgraph["edges"][n] for n in kedges}
        self.kedges_by_knodes = defaultdict(list)

        for e in kedges:
            self.kedges_by_knodes[
                tuple(sorted([kedges[e]["subject"], kedges[e]["object"]]))
            ].append(kedges[e])

        # find leaf-set qnodes
        degree = defaultdict(int)
        for edge in self.qgraph["edges"]:
            degree[self.qgraph["edges"][edge]["subject"]] += 1
            degree[self.qgraph["edges"][edge]["object"]] += 1
        self.leaf_sets = [
            node
            for node in self.qgraph["nodes"]
            if self.qgraph["nodes"][node].get("is_set", False) and degree[node] == 1
        ]

    def rank(self, answers, jaccard_like=False):
        """Generate a sorted list and scores for a set of subgraphs."""
        # get subgraph statistics
        answers_ = []
        scores_for_sort = []
        for answer in answers:
            answers_.append(self.score(answer, jaccard_like=jaccard_like))
            scores_for_sort.append(
                max(
                    analysis["score"]
                    for analysis in answer.get("analyses", [{"score": 0}])
                )
            )
        answers = [
            x
            for _, x in sorted(zip(scores_for_sort, answers), key=lambda pair: pair[0])
        ]
        return answers

    def score(self, answer, jaccard_like=False):
        """Compute answer score."""
        # answer is a list of dicts with fields 'id' and 'bound'
        r_node_ids, edges_all = self.get_rgraph(answer)

        for i_analysis, edges in enumerate(edges_all):
            # Identify Probes
            #################
            # Q Graph Connectivity Matrix
            q_node_ids = list(self.qgraph["nodes"].keys())
            n_q_nodes = len(q_node_ids)
            q_conn = np.full((n_q_nodes, n_q_nodes), 0)
            for e in self.qgraph["edges"].values():
                e_sub = q_node_ids.index(e["subject"])
                e_obj = q_node_ids.index(e["object"])
                if e_sub is not None and e_obj is not None:
                    q_conn[e_sub, e_obj] = 1

            # Determine probes based on connectivity
            node_conn = np.sum(q_conn, 0) + np.sum(q_conn, 1).T
            probe_nodes = []
            for conn in range(np.max(node_conn)):
                is_this_conn = node_conn == (conn + 1)
                probe_nodes += list(np.where(is_this_conn)[0])
                if len(probe_nodes) > 1:
                    break
            q_probes = list(combinations(probe_nodes, 2))

            # Converting qgraph inds to rgraph inds:
            qr_index = defaultdict(list)
            for node_id in r_node_ids[i_analysis]:
                qr_index[node_id[0]].append(r_node_ids[i_analysis].index(node_id))

            probes = []
            for probe in q_probes:
                left = qr_index[q_node_ids[probe[0]]]
                right = qr_index[q_node_ids[probe[1]]]
                for le in left:
                    for ri in right:
                        probes.append((le, ri))

            laplacian = self.graph_laplacian((r_node_ids[i_analysis], edges), probes)
            # If this still happens at this point it is because a probe has a problem
            if np.any(np.all(np.abs(laplacian) == 0, axis=0)):
                answer["analyses"][i_analysis]["score"] = 0
                continue

            score = np.exp(-kirchhoff(laplacian, probes))

            # fail safe to nuke nans
            score = score if np.isfinite(score) and score >= 0 else -1

            if jaccard_like:
                answer["analyses"][i_analysis]["score"] = score / (1 - score)
            else:
                answer["analyses"][i_analysis]["score"] = score
        return answer

    def graph_laplacian(self, rgraph, probes):
        """Generate graph Laplacian."""
        node_ids, edges = rgraph
        num_nodes = len(node_ids)
        # compute graph laplacian for this case while removing duplicate sources for each edge in the result graph
        
        # weight_dict = []
        # for subject_index in range(num_nodes):
        #     weight_dict_i = []
        #     for object_id in range(num_nodes):
        #         weight_dict_i.append({})
        #     weight_dict.append(weight_dict_i)
        
        weight_dict = defaultdict(lambda: defaultdict(\
            lambda: defaultdict( lambda: defaultdict(float))))
        for edge in edges:
            subject = edge["subject"]
            object = edge["object"]
            edge_weight = edge["weight"]

            for edge_source, edge_properties in edge_weight.items():
                for edge_property, edge_val in edge_properties.items():
                    weight_dict[subject][object][edge_source][edge_property] = \
                        max(weight_dict[subject][object][edge_source][edge_property], edge_val)

        qedge_qnode_ids = set(
            [frozenset((e["subject"], e["object"])) for e in self.qedge_by_id.values()]
        )
        laplacian = np.zeros((num_nodes, num_nodes))
        for i, sub_id_mapping in enumerate(node_ids):
            q_node_id_subject = sub_id_mapping[0]
            for j, obj_id_mapping in enumerate(node_ids):
                q_node_id_object = obj_id_mapping[0]

                edge_qnode_ids = frozenset((q_node_id_subject, q_node_id_object))

                # Set default weight (or 0 when edge is not a qedge)
                weight = (
                    self.DEFAULT_WEIGHT if edge_qnode_ids in qedge_qnode_ids else 0.0
                )

                for source, properties in weight_dict[sub_id_mapping][obj_id_mapping].items():
                    for property, source_w in properties.items():
                        source_weighted = source_w * source_weight(
                            source,
                            property,
                            source_weights=self.source_weights,
                            unknown_source_weight=self.unknown_source_weight,
                        )
                        if source_weighted == 1:
                            source_weighted = 0.99999999
                        weight = weight + -1 / (np.log(source_weighted))

                laplacian[i, j] += -weight
                laplacian[j, i] += -weight
                laplacian[i, i] += weight
                laplacian[j, j] += weight

        # Clean up Laplacian (remove extra nodes etc.)
        removal_candidate = np.all(np.abs(laplacian) == 0, axis=0)
        # Don't permit removing probes
        for probe in probes:
            removal_candidate[probe[0]] = False
            removal_candidate[probe[1]] = False

        keep = np.logical_not(removal_candidate)

        return laplacian[keep, :][:, keep]
    

    def get_rgraph(self, result):
        """Get "ranker" subgraph."""
        rnodes = set()
        redges = []
        dummy_ind = 0
        answer = copy.deepcopy(result)
        result_kg = {"node_ids": set(), "edge_ids": set()}

        for nb_id, nbs in answer.get("node_bindings", {}).items():
            for nb in nbs:
                n_id = nb.get("id", None)
                if n_id:
                    result_kg["node_ids"].add(n_id)

        nodes_list = []
        for anal in answer["analyses"]:
            anal_kg = copy.deepcopy(result_kg)

            for eb_id, ebs in anal["edge_bindings"].items():
                for eb in ebs:
                    e_id = eb.get("id", None)
                    if e_id:
                        anal_kg["edge_ids"].add(e_id)

            sg_ids = anal.get("support_graphs", [])
            for sg_id in sg_ids:
                sg = self.agraphs.get(sg_id, None)
                if sg:
                    sg_nodes = sg.get("nodes", [])
                    for sgn in sg_nodes:
                        anal_kg["node_ids"].add(sgn)

                    sg_edges = sg.get("edges", [])
                    for sge in sg_edges:
                        anal_kg["edge_ids"].add(sge)

            current_edge_ids = copy.deepcopy(anal_kg["edge_ids"])
            for edge_id in current_edge_ids:
                anal_kg = get_edge_support_kg(
                    edge_id, self.kgraph, self.agraphs, anal_kg
                )

            dummy_ind = 0
            extra_nodes = set()
            for e_id in anal_kg.get("edge_ids", []):
                # This shouldn't happen, but sometimes it does
                if e_id not in self.kgraph["edges"]:
                    logger.warning(f"Edge {e_id} not found in knowledge graph")
                    continue
                anal["edge_bindings"]["dummy_edge_" + str(dummy_ind)] = [{"id": e_id}]
                extra_nodes.add(self.kgraph["edges"][e_id]["subject"])
                extra_nodes.add(self.kgraph["edges"][e_id]["object"])
                dummy_ind += 1
            nodes_list.append(extra_nodes)

        # adds edges and nodes from support graphs into the answer edge bindings and node bindings
        # for i_analysis in range(len(answer.get('analyses',[]))):
        #     for sg in answer['analyses'][i_analysis].get('support_graphs',[]):
        #         for edge in self.agraphs[sg]['edges']:
        #             answer['analyses'][i_analysis]['edge_bindings']['dummy_edge_'+str(dummy_ind)] = [{'id':edge}]
        #             dummy_ind =+ 1

        # Checks if multiple nodes share node bindings
        rgraph_sets = [
            node
            for node in answer["node_bindings"]
            if len(answer["node_bindings"][node]) > 1
        ]
        # get list of nodes, and knode_map
        knode_map = defaultdict(set)
        for nb in answer["node_bindings"]:
            # get the query node binding entry
            qnode_id = nb
            knode_ids: list = []

            # get all the knowledge node entries
            for knode_id in answer["node_bindings"][nb]:
                knode_ids.append(knode_id["id"])

            for knode_id in knode_ids:
                rnode_id = (qnode_id, knode_id)
                rnodes.add(rnode_id)
                knode_map[knode_id].add(rnode_id)

                if qnode_id in rgraph_sets:
                    anchor_id = (f"{qnode_id}_anchor", "")
                    rnodes.add(anchor_id)
                    redges.append(
                        {
                            "weight": {"anchor_node": {"anchor_node": 1e9}},
                            "subject": rnode_id,
                            "object": anchor_id,
                        }
                    )
        rnodes = list(rnodes)
        # adding rnodes (for each results) to each nodes_list (for each analyses)
        [dump, knode_ids] = zip(
            *rnodes
        )  # don't want to duplicate rnodes already in there
        analyses_rnodes = []
        for i_analysis, nl in enumerate(nodes_list):
            dummy_node_count = 0
            anal_rnode = rnodes
            for n in nl:
                if n not in knode_ids:
                    anal_rnode.append(("dummy_node_" + str(dummy_node_count), n))
                    dummy_node_count += 1
            analyses_rnodes.append(anal_rnode)

        # for eb in answer['edge_bindings']:
        # qedge_id = eb
        # kedges = answer['edge_bindings'][eb]
        analysis_edges = []
        for i_analysis in range(len(answer.get("analyses", []))):
            # get "result" edges
            for qedge_id, kedge_bindings in answer["analyses"][i_analysis][
                "edge_bindings"
            ].items():
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
                    kedge = self.kedge_by_id[kedge_binding["id"]]
                    ksubject, kobject = kedge["subject"], kedge["object"]
                    try:
                        qedge = self.qedge_by_id[qedge_id]
                        qedge_nodes = qedge["subject"], qedge["object"]
                        pairs = [
                            pair
                            for pair in product(
                                [
                                    rnode
                                    for rnode in analyses_rnodes[i_analysis]
                                    if rnode[0] in qedge_nodes and rnode[1] == ksubject
                                ],
                                [
                                    rnode
                                    for rnode in analyses_rnodes[i_analysis]
                                    if rnode[0] in qedge_nodes and rnode[1] == kobject
                                ],
                            )
                            if pair[0][0] != pair[1][0]
                        ]
                    except KeyError:
                        # Support edges aren't bound to particular qnode_ids, so let's find all the places they can go
                        # set(tuple(sorted(pair)) for ...) prevents duplicate edges in opposite direction when kedge['subject'] == kedge['object']
                        pairs = set(
                            tuple(sorted(pair))
                            for pair in product(
                                [
                                    rnode
                                    for rnode in analyses_rnodes[i_analysis]
                                    if rnode[1] == ksubject
                                ],
                                [
                                    rnode
                                    for rnode in analyses_rnodes[i_analysis]
                                    if rnode[1] == kobject
                                ],
                            )
                            if pair[0][0] != pair[1][0]
                        )  # Prevents edges between nodes of the same qnode_id

                    for subject, object in pairs:
                        # get the weight from the edge binding
                        edge_vals = self.rank_vals.get(kedge_binding["id"], None)
                        if edge_vals is not None:
                            edge = {
                                "weight": {
                                    edge_vals["source"]: {
                                        k: v
                                        for k, v in edge_vals.items()
                                        if k != "source"
                                    }
                                },
                                "subject": subject,
                                "object": object,
                            }

                            redges.append(edge)
            analysis_edges.append(redges)

        return analyses_rnodes, analysis_edges


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
        # print(cols)
        return -np.inf

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
            if p.get("original_attribute_name", "") == "omnicorp_article_count":
                # save it
                omnicorp_article_count = p["value"]

                # no need to continue
                break

        # add the node d and count to the dict
        node_pubs.update({n: omnicorp_article_count})
    return node_pubs


def get_vals(edges, node_pubs, source_transfroamtion, unknown_source_transformation):
    # constant count of all publications
    all_pubs = 27840000

    # get the knowledge graph edges
    # edges = kgraph["edges"]
    edge_vals = {}
    # for each knowledge graph edge
    for edge in edges:
        # We are getting some results back (BTE?) that have "publications": ['PMID:1234|2345|83984']
        attributes = edges[edge].get("attributes", None)

        # init storage for the publications and their count
        publications = []
        num_publications = 0

        # Get source information
        edge_info_final = "unspecified"
        sources = edges[edge].get("sources", None)
        if sources is not None:
            for source in sources:
                if "primary_knowledge_source" in source.get("resource_role", None):
                    edge_info_final = source.get("resource_id", "unspecified")

        p_value = None
        if attributes is not None:
            # for each data attribute collect the needed params
            for attribute in attributes:
                # This picks up omnicorp
                if attribute.get("original_attribute_name", None) is not None:
                    # is this the publication list
                    if attribute["original_attribute_name"].startswith("publications"):
                        publications = attribute["value"]
                    # else is this the number of publications
                    elif attribute["original_attribute_name"].startswith(
                        "num_publications"
                    ):
                        num_publications = attribute.get("value", 0)
                    elif (
                        "p_value" in attribute["original_attribute_name"]
                        or "p-value" in attribute["original_attribute_name"]
                    ):
                        if isinstance(attribute["value"], list):
                            p_value = (
                                attribute["value"][0]
                                if len(attribute["value"]) > 0
                                else None
                            )
                        else:
                            p_value = attribute["value"]
                # This picks up Text Miner KP
                elif attribute["attribute_type_id"] == "biolink:supporting_document":
                    publications = attribute["value"]
                    if isinstance(publications, str):
                        publications = [publications]
                # This picks up how BTE returns pubs
                elif attribute["attribute_type_id"] == "biolink:publications":
                    publications = attribute["value"]
                elif (
                    "p_value" in attribute["attribute_type_id"]
                    or "p-value" in attribute["attribute_type_id"]
                ):
                    if isinstance(attribute["value"], list):
                        p_value = (
                            attribute["value"][0]
                            if len(attribute["value"]) > 0
                            else None
                        )
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
                            edge_info[attribute["attribute_type_id"]] = "unspecified"

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
                try:
                    subject_pubs = int(node_pubs[edges[edge]["subject"]])
                except:
                    subject_pubs = 0
                try:
                    object_pubs = int(node_pubs[edges[edge]["object"]])
                except:
                    object_pubs = 0
                # cast num_publications from json
                try:
                    num_publications = int(num_publications)
                except:
                    num_publications = 0

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
                edge_vals[edge]["p-value"] = source_sigmoid(
                    p_value,
                    edge_info_final,
                    "p-value",
                    source_transformation=source_transfroamtion,
                    unknown_source_transformation=unknown_source_transformation,
                )
            if literature_coocurrence is not None:
                edge_vals[edge]["literature_co-occurrence"] = source_sigmoid(
                    literature_coocurrence,
                    edge_info_final,
                    "literature_co-occurrence",
                    source_transformation=source_transfroamtion,
                    unknown_source_transformation=unknown_source_transformation,
                )
            if effective_pubs is not None:
                edge_vals[edge]["publications"] = source_sigmoid(
                    effective_pubs,
                    edge_info_final,
                    "publications",
                    source_transformation=source_transfroamtion,
                    unknown_source_transformation=unknown_source_transformation,
                )
            edge_vals[edge]["source"] = edge_info_final
    return edge_vals


def get_edge_support_kg(edge_id, kg, aux_graphs, edge_kg=None):
    if edge_kg is None:
        edge_kg = {"node_ids": set(), "edge_ids": set()}

    edge = kg["edges"].get(edge_id, None)
    if not edge:
        return edge_kg

    edge_attr = edge.get("attributes", None)
    if not edge_attr:
        return edge_kg

    edge_kg["edge_ids"].add(edge_id)

    # If we have edge attrs we might be adding new nodes
    sub = edge.get("subject", None)
    if sub:
        edge_kg["node_ids"].add(sub)

    obj = edge.get("object", None)
    if obj:
        edge_kg["node_ids"].add(obj)

    for attr in edge_attr:
        attr_type = attr.get("attribute_type_id", None)
        if attr_type == "biolink:support_graphs":
            # We actually have a biolink support graph
            more_support_graphs = attr.get("value", [])
            for sg_id in more_support_graphs:
                sg = aux_graphs.get(sg_id, None)
                if not sg:
                    continue

                sg_edges = sg.get("edges", [])
                sg_nodes = sg.get("nodes", [])
                for sgn in sg_nodes:
                    edge_kg["node_ids"].add(sgn)

                for add_edge_id in sg_edges:
                    try:
                        add_edge = kg["edges"][add_edge_id]
                    except KeyError:
                        # This shouldn't happen, but it's defending against some malformed TRAPI
                        continue

                    # Get this edge and add it to the edge_kg
                    edge_kg["edge_ids"].add(add_edge_id)

                    add_edge_sub = add_edge.get("subject", None)
                    if add_edge_sub:
                        edge_kg["node_ids"].add(add_edge_sub)

                    add_edge_object = add_edge.get("object", None)
                    if add_edge_object:
                        edge_kg["node_ids"].add(add_edge_object)

                    edge_kg = get_edge_support_kg(add_edge_id, kg, aux_graphs, edge_kg)

    return edge_kg
