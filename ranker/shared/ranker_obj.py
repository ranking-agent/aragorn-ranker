"""ROBOKOP ranking."""

import copy
import logging
from collections import defaultdict
from itertools import combinations, product

import numpy as np

from ranker.shared.sources import get_profile, get_source_sigmoid, get_source_weight, get_base_weight

logger = logging.getLogger(__name__)


class Ranker:
    """Ranker."""

    DEFAULT_WEIGHT = -1/np.log(1e-2)

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
            base_weights
        ) = get_profile(profile)
        self.source_weights = source_weights
        self.unknown_source_weight = unknown_source_weight
        self.source_transformation = source_transformation
        self.unknown_source_transformation = unknown_source_transformation
        self.base_weights = base_weights

        # There are caches stored for this message
        # Initialized here. 
        # These are used to find numeric values of edges
        self.node_pubs = dict()
        self.edge_values = dict()
        # self.rank_vals = get_vals(
        #     self.kgraph["edges"],
        #     self.node_pubs,
        #     self.source_transformation,
        #     self.unknown_source_transformation,
        # )

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
                "score": score,
                "r_graph": r_graph,
                "probes": probes,
                "edges": {e_info[2]:self.kgraph["edges"][e_info[2]] for e_info in r_graph["edges"]},
                "laplacian": laplacian,
                "probe_inds": probe_inds
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
        # Make a dictionary of edge subject, predicate, source , properties.
        #
        # We will also keep track of a weighted version of the edge values
        # these are influenced by the profile
        edge_values_mat = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for edge in edges:
            # Recall that edge is a tuple
            # This will contain
            # (r_node_subject_id, r_node_subject_id, k_edge_id)
            r_subject = edge[0]
            r_object = edge[1]
            k_edge_id = edge[2]

            # The edge weight can be found via lookup
            # With a little messaging
            edge_vals = self.get_edge_values(k_edge_id)
            
            edge_values_mat[r_subject][r_object][k_edge_id] = edge_vals
            # This enforces symmetry in edges/wires
            edge_values_mat[r_object][r_subject][k_edge_id] = edge_vals # Enforce symmetry

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
                
                for edge_id, edge in edge_values_mat[sub_r_node_id][obj_r_node_id].items():
                    for source, properties in edge.items():
                        for property, values in properties.items():
                            w = values['weight']
                            
                            if w >= 1: # > as an emergency
                                w = 0.9999999 # 1 causes numerical issues
                            
                            # -1 / np.log(source_weighted) is our mapping from a [0,1]
                            # weight to an admittance. 
                            # These admittances just add since they are all in parallel
                            # This is equivalent to a noisy or.
                            weight = weight + -1 / (np.log(w))

                weight_mat[i, j] += weight # For debugging

        # Using weight_mat you can calculated the laplacian
        # We could do this in the loop above, but let's be explicit
        weight_row_sums = np.sum(weight_mat,axis=1)
        laplacian = -1 * weight_mat.copy()
        for i in range(num_nodes):
            laplacian[i, i] = weight_row_sums[i]

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
            "edge_values": edge_values_mat,
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

    def get_omnicorp_node_pubs(self, node_id): 
        """
        Find and return the omnicorp publication counts attached to each node.
        This method will also cache the result for this message
        """
        
        # Check cache
        if node_id in self.node_pubs:
            return self.node_pubs[node_id]
        
        # Not in the cache, make sure it's a valid node
        node = self.kgraph['nodes'].get(node_id)
        if not node:
            # Should we error here or just cache 0?
            # I think error
            raise KeyError(f"Invalid node ID {node_id}")
        
        # Extract the node information we are interested in

        # Parse the node attributes to find the publication count
        omnicorp_article_count = 0
        attributes = node.get('attributes',[])
        
        # Look through attributes and check for the omnicorp edges
        for p in attributes:
            # is this what we are looking for
            # Over time this has changed it's name
            # num_publications is currently in use (2024-03)
            # but for historical sake we keep the old name
            if p.get("original_attribute_name", "") in ["omnicorp_article_count", "num_publications"]:
                omnicorp_article_count = p["value"]
                break # There can be only one
        
        # Safely parse
        try:
            omnicorp_article_count = int(omnicorp_article_count)
        except:
            omnicorp_article_count = 0

        # Cache it
        self.node_pubs[node_id] = omnicorp_article_count

        return omnicorp_article_count
    
    def get_edge_values(self, edge_id):
        """
        This transforms all edge attributes into values that can be used for ranking
        This does not consider all attributes, just the ones that we can currently handle.
        If we want to handle more things we need to add more cases here.
        This will also cache the result for this message
        """

        # literature co-occurrence assumes a global number of pubs in it's calculation
        # This is a constant/param and could potentially be included in a profile
        TOTAL_PUBS = 27840000

        # Check cache
        if edge_id in self.edge_values:
            return self.edge_values[edge_id]
        
        # Not in the cache, make sure it's a valid node
        edge = self.kgraph['edges'].get(edge_id)
        if not edge:
            # Should we error here or just cache empty?
            # I think error
            raise KeyError(f"Invalid edge ID {edge_id}")
        
        # Extract the edge information we are interested in

        # Get edge source information this is looking for primary_knowledge_source
        edge_source = "unspecified"
        for source in edge.get("sources", []):
            if "primary_knowledge_source" == source.get("resource_role", None):
                edge_source = source.get("resource_id", "unspecified")
                break # There can be only one
        
        # We find literature co-occurance edges via predicate
        edge_pred = edge.get("predicate", '')

        # Currently we use three types of information
        # Init storage for the values we may find
        usable_edge_attr = {
            "publications": [],
            "num_publications": 0,
            "literature_coocurrence": None,
            "p_value": None
        }

        # Look through attributes and 
        for attribute in edge.get("attributes", []):
            orig_attr_name = attribute.get("original_attribute_name", None)
            attr_type_id = attribute.get("attribute_type_id", None)

            # We will look at both the original_attribute_name and the
            # attribute_type_id. The attribute_type_id is the real method
            # But we should maintain some degree of backwards compatibility
            
            # Publications
            if orig_attr_name == "publications" or \
                attr_type_id == "biolink:supporting_document" or \
                attr_type_id == "biolink:publications":
                
                # Parse pubs to handle all the cases we have observed
                pubs = attribute.get("value", [])
                
                if isinstance(pubs, str):
                    pubs = [pubs]

                # Attempt to parse pubs incase it has string lists
                if len(pubs) == 1:
                    if "|" in pubs[0]:
                        # "publications": ['PMID:1234|2345|83984']
                        pubs = pubs[0].split("|")
                    elif "," in pubs[0]:
                        # "publications": ['PMID:1234,2345,83984']
                        pubs = pubs[0].split(",")

                usable_edge_attr["publications"] = pubs
                usable_edge_attr["num_publications"] = len(pubs)
            
            # P-Values
            if "p_value" in orig_attr_name or "p-value" in orig_attr_name or \
                "p_value" in attr_type_id or "p-value" in attr_type_id:
                
                p_value = attribute.get("value", None)

                # Some times the reported p_value is a list like [p_value]
                if isinstance(attribute["value"], list):
                    p_value = (p_value[0] if len(p_value) > 0 else None)

                usable_edge_attr["p_value"] = p_value

            # Literature Co-occurrence actually uses the num_publications found above
            # So we make sure we do it last.
            if edge_pred == "biolink:occurs_together_in_literature_with" and \
                attr_type_id == "biolink:has_count":

                # We assume this is from a literature co-occurrence source like omnicorp
                np = attribute.get("value", 0)
                # Parse strings safely
                try:
                    np = int(np)
                except:
                    np = 0

                subject_pubs = self.get_omnicorp_node_pubs(edge["subject"])
                object_pubs = self.get_omnicorp_node_pubs(edge["object"])

                # Literature co-occurrence score
                cov = (np / TOTAL_PUBS) - (subject_pubs / TOTAL_PUBS) * (
                    object_pubs / TOTAL_PUBS
                )
                cov = max((cov, 0.0))
                usable_edge_attr['literature_coocurrence'] = cov * TOTAL_PUBS
            # else:
            #     # Every other edge has an assumed publication of 1
            #     usable_edge_attr['num_publications'] += 1
            
        # At this point we have all of the information extracted from the edge
        # We have have looked through all attributes and filled up usable_edge_attr

        this_edge_vals = defaultdict(dict)
        base_weight = get_base_weight(edge_source, self.base_weights)
        this_edge_vals[edge_source]["base_weight"] = {
            "weight": base_weight
        }
        if usable_edge_attr["p_value"] is not None:
            property_w = get_source_sigmoid(
                usable_edge_attr["p_value"],
                edge_source,
                "p-value",
                self.source_transformation,
                self.unknown_source_transformation
            )
            source_w = get_source_weight(
                edge_source,
                "p-value",
                self.source_weights,
                self.unknown_source_weight
            )
            
            this_edge_vals[edge_source]["p_value"] = {
                "value": usable_edge_attr["p_value"],
                "property_weight": property_w,
                "source_weight": source_w,
                "weight": property_w * source_w
            }

        if usable_edge_attr['num_publications']:
            property_w = get_source_sigmoid(
                usable_edge_attr['num_publications'],
                edge_source,
                "publications",
                self.source_transformation,
                self.unknown_source_transformation,
            )

            source_w = get_source_weight(
                edge_source,
                "publications",
                self.source_weights,
                self.unknown_source_weight
            )

            this_edge_vals[edge_source]["publications"] = {
                "value": usable_edge_attr["num_publications"],
                "property_weight": property_w,
                "source_weight": source_w,
                "weight": property_w * source_w
            }

        if usable_edge_attr['literature_coocurrence'] is not None:
            
            property_w = get_source_sigmoid(
                usable_edge_attr['literature_coocurrence'],
                edge_source,
                "literature_co-occurrence",
                self.source_transformation,
                self.unknown_source_transformation,
            )

            source_w = get_source_weight(
                edge_source,
                "literature_co-occurrence",
                self.source_weights,
                self.unknown_source_weight
            )

            this_edge_vals[edge_source]["literature_coocurrence"] = {
                "value": usable_edge_attr["literature_coocurrence"],
                "property_weight": property_w,
                "source_weight": source_w,
                "weight": property_w * source_w
            }
            
        # Cache it
        self.edge_values[edge_id] = this_edge_vals

        return this_edge_vals

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
