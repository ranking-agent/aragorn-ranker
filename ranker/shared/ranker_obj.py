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

        # Decompose message
        self.kgraph = message.get("knowledge_graph", {"nodes": {}, "edges": {}})
        self.qgraph = message.get("query_graph", {"nodes": {}, "edges": {}})
        self.agraphs = message.get("auxiliary_graphs", {})
        
        # Apply profile
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

        # Find numeric values of edges
        # This could be smarter and fetched and memoized
        # rank_vals (get_vals) is used once in graph_laplacian
        # node_pubs is only used by rank_vals
        # This could be a big time savings for large messages
        self.node_pubs = get_node_pubs(self.kgraph)
        self.rank_vals = get_vals(
            self.kgraph["edges"],
            self.node_pubs,
            self.source_transformation,
            self.unknown_source_transformation,
        )

    def rank(self, answers, jaccard_like=False):
        """Generate a sorted list and scores for a set of subgraphs."""
        # get subgraph statistics
        answers_ = []
        scores_for_sort = []
        for answer in answers:
            scored_answer, _ = self.score(answer, jaccard_like=jaccard_like)
            answers_.append(scored_answer)
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

    def probes(self):
        
        # Identify Probes
        #################
        # Q Graph Connectivity Matrix
        q_node_ids = list(self.qgraph["nodes"].keys()) # Need to preserve order!
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

        # Convert probes back to q_node_ids
        probes = [(q_node_ids[p[0]],q_node_ids[p[1]]) for p in q_probes]

        return probes
    
    def score(self, answer, jaccard_like=False):
        """Compute answer score."""
        
        # Probes used for scoring are based on the q_graph
        # This will be a list of q_node_id tuples
        probes = self.probes()

        # The r_graph is all of the information we need to score each analysis
        # This includes walking through all of the support graphs
        # And organizing nodes and edges into a more manageable form scoring
        # There is some repeated work accross analyses so we calculate all r_graphs
        # at once
        r_gaphs = self.get_rgraph(answer)

        # For each analysis we have a unique r_graph to score
        analysis_details = []
        for i_analysis, r_graph in enumerate(r_gaphs):
            # First we calculate the graph laplacian
            # The probes are needed to make sure we don't remove anything
            # that we actually wanted to use for scoring
            laplacian, probe_inds, laplacian_details = self.graph_laplacian(r_graph, probes)

            # For various reasons (malformed responses typicall), we might have a 
            # weird laplacian. We already checked and tried to clean up above
            # If this still happens at this point it is because a probe has a problem
            if np.any(np.all(np.abs(laplacian) == 0, axis=0)):
                answer["analyses"][i_analysis]["score"] = 0
                continue
 
            # Once we have the graph laplacian we can find the effective resistance
            # Between all of the probes
            # The exp(-1 * .) here converts us back to normalized space
            score = np.exp(-kirchhoff(laplacian, probe_inds))

            # Fail safe to get rid of NaNs.
            score = score if np.isfinite(score) and score >= 0 else -1

            if jaccard_like:
                answer["analyses"][i_analysis]["score"] = score / (1 - score)
            else:
                answer["analyses"][i_analysis]["score"] = score

            # Package up details
            this_analysis_details = {
                "r_graph": r_graph,
                "laplacian": laplacian,
                "probe_inds": probe_inds,
                "score": score,
                "edges": {e_info[2]:self.kgraph["edges"][e_info[2]] for e_info in r_graph["edges"]}
            }
            this_analysis_details.update(laplacian_details)

            analysis_details.append(this_analysis_details)

        return answer, analysis_details

    def graph_laplacian(self, r_graph, probes):
        """Generate graph Laplacian."""
        
        nodes = list(r_graph['nodes']) # Must convert to list
        edges = list(r_graph['edges']) # Must have consistent order
        
        # The graph laplacian will be a square matrix
        # If all goes well it will be len(nodes) by len(nodes)
        num_nodes = len(nodes)

        # For each edge in the answer graph
        # Make a dictionary of edge source / properties.
        # For the case where there are redundant edges,
        # same subject, object, source, property type,
        # Take the max of the property values
        # This might happen if two KPs have the same underlying data sources
        # But for some reason return different publication counts        
        weight_dict = defaultdict(lambda: defaultdict(\
            lambda: defaultdict( lambda: defaultdict(float))))
        
        # We will also keep track of this weighted version of the weight dictionary
        # but this is for a diagnostic output
        weight_dict_profile = defaultdict(lambda: defaultdict(\
            lambda: defaultdict( lambda: defaultdict(float))))
        edge_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for edge in edges:
            # Recall that edge is a tuple
            # This will contain
            # (r_node_subject_id, r_node_subject_id, k_edge_id)
            r_subject = edge[0]
            r_object = edge[1]
            k_edge_id = edge[2]

            # The edge weight can be found via lookup
            # With a little messaging
            edge_vals = self.rank_vals.get(k_edge_id, None)
            if edge_vals is None:
                # Wacky edge
                continue

            edge_weight = {
                edge_vals["source"]: {
                    k: v
                    for k, v in edge_vals.items()
                    if k != "source"
                }
            }
            edge_dict[r_subject][r_object][k_edge_id] = edge_weight

            for edge_source, edge_properties in edge_weight.items():
                for edge_property, edge_val in edge_properties.items():
                    weight_dict[r_subject][r_object][edge_source][edge_property] = \
                        max(weight_dict[r_subject][r_object][edge_source][edge_property], edge_val)
                    weight_dict[r_object][r_subject][edge_source][edge_property] = \
                        max(weight_dict[r_object][r_subject][edge_source][edge_property], edge_val)

        # Make a set of all subject object q_node_ids that have q_edges
        qedge_qnode_ids = set(
            [frozenset((e["subject"], e["object"])) for e in self.qgraph["edges"].values()]
        )

        # Now go through these edges
        # Turn each value into an edge weight
        # Then calculate the graph laplacian
        laplacian = np.zeros((num_nodes, num_nodes))
        weight_mat = np.zeros((num_nodes, num_nodes)) # For debugging
        for i, sub_r_node_id in enumerate(nodes):
            for j, obj_r_node_id in enumerate(nodes):
                # If these r_nodes correspond to q_nodes and there is a q_edge between
                # Then we set a default weight for this weight
                # Otherwise we set the default weight to 0
                #
                # This ensures that every bound q_edge at least counts for something
                edge_qnode_ids = frozenset((sub_r_node_id, obj_r_node_id))
                weight = (
                    self.DEFAULT_WEIGHT if edge_qnode_ids in qedge_qnode_ids else 0.0
                )

                # Map each of these weights according to the profile
                # Then parallel combine all of the weights between this subject and object
                for source, properties in weight_dict[sub_r_node_id][obj_r_node_id].items():
                    for property, source_w in properties.items():
                        source_weighted = source_w * source_weight(
                            source,
                            property,
                            source_weights=self.source_weights,
                            unknown_source_weight=self.unknown_source_weight,
                        )

                        weight_dict_profile[sub_r_node_id][obj_r_node_id][property] = source_weighted

                        if source_weighted >= 1: # > as an emergency
                            source_weighted = 0.9999999 # 1 causes numerical issues so we want basically 1
                        weight = weight + -1 / (np.log(source_weighted))

                weight_mat[i, j] += weight # For debugging

                laplacian[i, j] += -weight
                laplacian[j, i] += -weight
                laplacian[i, i] += weight
                laplacian[j, j] += weight
        
        # Using weight_mat you can calculated the laplacian, however we did this above.
        # weight_row_sums = np.sum(weight_mat,axis=1)
        # laplacian = -1 * weight_mat.copy()
        # for i in range(num_nodes):
        #     laplacian[i, i] = weight_row_sums[i]

        # Clean up Laplacian (remove extra nodes etc.)
        # Sometimes, mostly because of a bug of some kind,
        # There will be rows of all zeros in the laplacian.
        # This will cause numerical issues.
        # We can remove these, as long as they aren't probes.
        removal_candidate = np.all(np.abs(laplacian) == 0, axis=0)
        # Don't permit removing probes
        for p in probes:
            p_i = (nodes.index(p[0]), nodes.index(p[1]))
            removal_candidate[p_i[0]] = False
            removal_candidate[p_i[1]] = False

        keep = np.logical_not(removal_candidate)
        kept_nodes = [n for i, n in enumerate(nodes) if keep[i]]

        # Convert probes to new laplacian inds
        probe_inds = [(kept_nodes.index(p[0]), kept_nodes.index(p[1])) for p in probes]


        details = {
            "edge_dict": edge_dict,
            "weight_dict": weight_dict,
            "weight_dict_profile": weight_dict_profile,
            "weight_mat": weight_mat
        }
        
        return laplacian[keep, :][:, keep], probe_inds, details
    

    def get_rgraph(self, result):
        """Get "ranker" subgraph."""
        answer = copy.deepcopy(result)
        
                # # Super nodes are any q_nodes that contain multiple bindings
                # # These will be treated differently throughout scoring
                # super_nodes = defaultdict(set)
                # for node in answer["node_bindings"]:
                #     if len(answer["node_bindings"][node]) > 1:
                #         super_nodes[node] = set([nb['id'] for nb in answer["node_bindings"][node]])
        
        # All analyses share some common r_graph nodes. We can make those now
        r_graph_shared = dict()
        r_graph_shared['nodes'] = set()
        r_graph_shared['nodes_map'] = defaultdict(list)
        for nb_id, nbs in answer['node_bindings'].items():
            r_graph_shared['nodes'].add(nb_id)
            for nb in nbs:
                r_graph_shared['nodes_map'][nb['id']].append(nb_id)

        # Build the results KG for each analysis
        
        # The nodes for the results KG are the same for all analyses
        # We can populate these node_ids by walking through all node bindings
        result_kg_shared = {"node_ids": set(), "edge_ids": set()}
        for nb_id, nbs in answer.get("node_bindings", {}).items():
            for nb in nbs:
                n_id = nb.get("id", None)
                if n_id:
                    result_kg_shared["node_ids"].add(n_id)

        # For each analysis we need to build a KG of all nodes and edges
        analysis_r_graphs = []
        for anal in answer["analyses"]:
            # Copy this list of globally bound nodes
            anal_kg = copy.deepcopy(result_kg_shared)

            # Walk and find all edges in this analysis
            for eb_id, ebs in anal["edge_bindings"].items():
                for eb in ebs:
                    e_id = eb.get("id", None)
                    if e_id:
                        anal_kg["edge_ids"].add(e_id)

            # Parse through all support graphs used in this analysis
            sg_ids = anal.get("support_graphs", [])
            for sg_id in sg_ids:
                sg = self.agraphs.get(sg_id, None)
                if sg:
                    # We found the referenced support graph 
                    # Add in the corresponding nodes
                    sg_nodes = sg.get("nodes", [])
                    for sgn in sg_nodes:
                        anal_kg["node_ids"].add(sgn)

                    # Add in the corresponding edges
                    sg_edges = sg.get("edges", [])
                    for sge in sg_edges:
                        anal_kg["edge_ids"].add(sge)

            # Now we need to go through all of the edges we have found thus far
            # We need to find any additional support graphs
            # We will do that with this recursive function call
            # Since results_kg uses sets, things will not get duplicated
            current_edge_ids = copy.deepcopy(anal_kg["edge_ids"])
            for edge_id in current_edge_ids:
                anal_kg = get_edge_support_kg(
                    edge_id, self.kgraph, self.agraphs, anal_kg
                )

            # At this point each analysis now has a complete anal_kg
            # This is the complete picture of all nodes and edges used by this analysis
            # This includes everything from all support graphs (recursively)

            # To make things simpler below it is helpful if will build a complete list
            # of additional nodes and edges that have been added as part of support
            anal["support_nodes"] = copy.deepcopy(anal_kg['node_ids'])
            anal["support_edges"] = copy.deepcopy(anal_kg['edge_ids'])
            for eb_id, ebs in anal["edge_bindings"].items():
                for eb in ebs:
                    e_id = eb.get("id", None)
                    if e_id and e_id in anal["support_edges"]:
                        anal["support_edges"].remove(e_id)

            for nb_id, nbs in answer['node_bindings'].items():
                for nb in nbs:
                    n_id = nb.get("id", None)
                    if n_id and n_id in anal["support_nodes"]:
                        anal["support_nodes"].remove(n_id)

            # It is also convenient to have a list of all edges that were bound
            anal["bound_edges"] = anal_kg['edge_ids'] - anal["support_edges"]

            # We need to build the r_graph which is a little different than the analysis graph
            # In the list of nodes in the analysis we need to consider the specific node bindings
            # For example, it is possible to use a k_node in multiple bindings to different q_nodes
            # the r_graph makes "r_nodes" for each q_node
            #
            # Any additional support nodes are added accordingly to the r_graph as individual r_nodes
            #
            # Then we need to include all edges but have them point at the correct r_nodes
            # We need to reroute them accordingly by looking up the origin k_node ids in the nodes_map

            # First we copy over the shared nodes
            anal_r_graph = copy.deepcopy(r_graph_shared)
            
            # Add in support nodes to the r_graph
            # We will make an r_graph node
            # And a mapping to that r_graph node
            for sn in anal["support_nodes"]:
                r_graph_node_id = f"support_node_{sn}"
                anal_r_graph['nodes'].add(r_graph_node_id)
                anal_r_graph['nodes_map'][sn].append(r_graph_node_id)
            
            # For each edge we need to find the corresponding r_nodes
            # for the subject and object
            # We will use a tuple for these as a sparse matrix type of thing
            # Bound edges will go between non support nodes
            anal_r_graph['edges'] = set()
            for e_id in anal["bound_edges"] | anal["support_edges"]:
                e = self.kgraph['edges'].get(e_id)
                if not e:
                    logger.warning(f"Edge {e_id} not found in knowledge graph")
                    continue
                subject_id = e['subject']
                subject_r_ids = anal_r_graph['nodes_map'][subject_id]

                object_id = e['object']
                object_r_ids = anal_r_graph['nodes_map'][object_id]

                for r_sub in subject_r_ids:
                    for r_obj in object_r_ids:
                        anal_r_graph['edges'].add((r_sub, r_obj, e_id))

            # Build a list of these for each analysis
            analysis_r_graphs.append(anal_r_graph)

        return analysis_r_graphs


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
