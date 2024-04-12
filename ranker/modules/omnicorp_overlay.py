"""Literature co-occurrence support."""
import heapq
import logging
import os
import asyncio

from uuid import uuid4
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from fastapi.responses import JSONResponse
from ranker.shared.cache import Cache
from ranker.shared.util import batches, create_log_entry
from reasoner_pydantic import Response as PDResponse

from typing import List, Dict, Tuple, Set

logger = logging.getLogger(__name__)

CACHE_HOST = os.environ.get("CACHE_HOST", "localhost")
CACHE_PORT = os.environ.get("CACHE_PORT", "6380")
CACHE_DB = os.environ.get("CACHE_DB", "0")
CACHE_PASSWORD = os.environ.get("CACHE_PASSWORD", "")


async def add_node_pmid_counts(kgraph, counts):
    for node_id in kgraph["nodes"]:
        if node_id in counts:
            count = counts[node_id]
        else:
            count = 0

        # add omnicorp_article_count to nodes in networkx graph
        attribute = {
            "original_attribute_name": "omnicorp_article_count",
            "attribute_type_id": "biolink:has_count",
            "value": count,
            "value_type_id": "EDAM:data_0006",
        }

        # if there is no attributes array add one.  We used exclude_None=True so we don't need to check for it here.
        if "attributes" not in kgraph["nodes"][node_id]:
            kgraph["nodes"][node_id]["attributes"] = []

        # save the attributes
        kgraph["nodes"][node_id]["attributes"].append(attribute)

async def add_shared_pmid_counts(
    message,
    values,
    pair_to_answer,
):
    """Count PMIDS shared by a pair of nodes and create a new support edge."""
    kgraph = message["knowledge_graph"]
    aux_graphs = message["auxiliary_graphs"]
    answers = message["results"]
    support_idx = 0
    for pair, publication_count in values.items():
        if publication_count == 0:
            continue
        uid = str(uuid4())
        kgraph["edges"].update(
            {
                uid: {
                    "predicate": "biolink:occurs_together_in_literature_with",
                    "attributes": [
                        {
                            "original_attribute_name": "num_publications",
                            "attribute_type_id": "biolink:has_count",
                            "value_type_id": "EDAM:data_0006",
                            "value": publication_count,
                        },
                        {
                            "attribute_type_id": "biolink:agent_type",
                            "value": "statistical_association_pipeline"
                        },
                        {
                            "attribute_type_id": "biolink:knowledge_level",
                            "value": "statistical_association"
                        }
                    ],
                    "sources": [{
                            "resource_id": "infores:omnicorp",
                            "resource_role": "primary_knowledge_source",
                    }],
                    "subject": pair[0],
                    "object": pair[1],
                }
            }
        )

        # pair_to_answer is a dictionary of pairs to tuples.
        # Each tuple is a pair of (answer_idx, analysis_idx)
        for answer_idx, analysis_idx in pair_to_answer[pair]:
            analysis = answers[answer_idx]["analyses"][analysis_idx]
            # see if the analysis has an omnicoprot support graph
            if "support_graphs" not in analysis:
                analysis["support_graphs"] = []
            omnisupport = None
            for sg in analysis["support_graphs"]:
                if sg.startswith("OMNICORP_support_graph"):
                    omnisupport = sg
                    break
            if omnisupport is None:
                omnisupport = f"OMNICORP_support_graph_{support_idx}"
                support_idx += 1
                analysis["support_graphs"].append(omnisupport)
            if omnisupport not in aux_graphs:
                aux_graphs[omnisupport] = { "edges": [] }
            aux_graphs[omnisupport]["edges"].append(uid)


def create_node2pairs(pairs: Set[Tuple]) -> Dict[str, List[Tuple]] :
    """Create a dictionary of node to pairs."""
    node2pairs = defaultdict(list)
    for pair in pairs:
        for node in pair:
            node2pairs[node].append(pair)
    return node2pairs

def make_key(x,node_indices):
    """x is a tuple of curies.
    node_indices is a dictionary of node to index.
    return a string "index1_index2" where index1 < index2.
    """
    i1 = node_indices[x[0]]
    i2 = node_indices[x[1]]
    if i1 < i2:
        return f"{i1}_{i2}"
    else:
        return f"{i2}_{i1}"

async def query(request: PDResponse):
    """Add support to message.

    Add support edges to knowledge_graph and bindings to results.
    """
    logger.info("Start omnicorp")
    # get the debug environment variable
    debug = os.environ.get("DEBUG_TIMING", "False")

    if debug == "True":
        dt_start = datetime.now()

    dt_1 = datetime.now()
    in_message = request.dict(exclude_none=True, exclude_unset=True)
    dt_2 = datetime.now()
    logger.info(f"convert in message to dict: {dt_2 - dt_1}")

    # save the logs for the response (if any)
    if "logs" not in in_message or in_message["logs"] is None:
        in_message["logs"] = []
    else:
        # these timestamps are causing json serialization issues
        # so here we convert them to strings.
        for log in in_message["logs"]:
            log["timestamp"] = str(log["timestamp"])

    # init the status code
    status_code: int = 200

    message = in_message["message"]

    qgraph = message["query_graph"]
    kgraph = message["knowledge_graph"]
    answers = message["results"]
    if "auxiliary_graphs" not in message:
        message["auxiliary_graphs"] = {}

    # get cache if possible
    try:
        cache = Cache(
            redis_host=CACHE_HOST,
            redis_port=CACHE_PORT,
            redis_db=CACHE_DB,
            redis_password=CACHE_PASSWORD,
        )
    except Exception as e:
        logger.exception(e)
        cache = None

    dt_start = datetime.now()
    redis_batch_size = 1000

    try:
        # get all node supports.
        start_node_time = datetime.now()
        # first, get the publication counts for all nodes in the kgraph
        keys = list(kgraph["nodes"].keys())

        node_pub_counts = {}
        node_indices = {}
        for batch in batches(keys, redis_batch_size):
            results = cache.curie_query(batch)
            for curie, result in results.items():
                if len(result) == 0:
                    continue
                node_pub_counts[curie] = result["pmc"]
                node_indices[curie] = int(result["index"])
        await add_node_pmid_counts(kgraph, node_pub_counts)
        end_node_time = datetime.now()
        logger.info(f"Node time: {end_node_time - start_node_time}")

        start_pair_time = datetime.now()
        # which qgraph nodes are sets?
        qgraph_setnodes = set(
            [
                n
                for n in qgraph["nodes"]
                if (
                    ("is_set" in qgraph["nodes"][n])
                    and (qgraph["nodes"][n]["is_set"])
                )
            ]
        )

        #Now we want to find the publication count for every pair.   But: we only want to do that for pairs that
        # are part of the same answer
        t1 = datetime.now()
        pair_to_answer = await generate_curie_pairs(answers, qgraph_setnodes, node_pub_counts, message)
        t2 = datetime.now()
        logger.info(f"generate_curie_pairs time: {t2 - t1}. Number of pairs: {len(pair_to_answer)}")

        # get all pair supports
        keypairs = {make_key(x,node_indices):x for x in pair_to_answer.keys()}
        inputkeys = list(keypairs.keys())

        values = {}
        for batch in batches(inputkeys, redis_batch_size):
            q_start = datetime.now()
            results = cache.shared_count_query(batch)
            q_end = datetime.now()
            #logger.info(f"Query time: {q_end - q_start}")
            for input,output in results.items():
                if output is not None:
                    curie_pair = keypairs[input]
                    try:
                        values[curie_pair] = int(output)
                    except Exception as e:
                        values[curie_pair] = 0
        await add_shared_pmid_counts(message,values,pair_to_answer)
        end_pair_time = datetime.now()
        logger.info(f"Pair time: {end_pair_time - start_pair_time}")

        # load the new results into the response
        message["knowledge_graph"] = kgraph
        message["results"] = answers

    except Exception as e:
        # put the error in the response
        status_code = 500
        logger.exception(f"Aragorn-ranker/omnicorp exception {e}")
        # save any log entries
        # in_message['logs'].append(create_log_entry(f'Exception: {str(e)}', 'ERROR'))

    if debug == "True":
        diff = datetime.now() - dt_start
        in_message["logs"].append(
            create_log_entry(
                f"End of omnicorp overlay processing. Time elapsed: {diff.seconds} seconds",
                "DEBUG",
            )
        )

    # return the result to the caller
    logger.info("Omnicorp complete. Returning.")
    return JSONResponse(content=in_message, status_code=status_code)


async def generate_curie_pairs(answers, qgraph_setnodes, node_pub_counts, message):
    # Generate a set of pairs of node curies
    # if we don't have a node in node_pub_counts, we don't need to add it to the pairs to check.
    pair_to_answer = defaultdict(set)  # a map of node pairs to answers
    for ans_idx, answer_map in enumerate(answers):

        # Get all nodes that are not part of sets and densely connect them
        # can be str (not a set) or list (could be a set or not a set)
        nonset_nodes = []
        setnodes = {}

        #What counts as a node in the answer for TRAPI 1.4?
        #It can be in the node_bindings
        #Or, you can go into each analysis, go to the edges in the edge binding, and some of those many have
        # an auxiliary supporting graph, which has edges.  The edges have nodes. Those nodes count.
        # for the ones in this case, the support graph goes in the analysis, and involves both the
        # bound nodes and the analysis nodes.

        # node binding results is now a dict containing dicts that contain a list of dicts.
        for nb in answer_map["node_bindings"]:
            if nb in qgraph_setnodes:
                setnodes[nb] = [
                    node["id"] for node in answer_map["node_bindings"][nb]
                ]
            else:
                if len(answer_map["node_bindings"][nb]) != 0:
                    nonset_nodes.extend(
                        [x["id"] for x in answer_map["node_bindings"][nb]]
                    )

        for analysis_idx, analysis in enumerate(answer_map["analyses"]):
            new_nonset_nodes = set()
            # find the knowledge edges that are bound in the analysis
            relevant_kedge_id_lists = [ [x["id"] for x in eb] for eb in analysis["edge_bindings"].values()]
            relevant_kedge_ids = [x for el in relevant_kedge_id_lists for x in el]
            #for bound knowledge edges, find their supporting graphs
            auxgraph_ids = []
            for kedge_id in relevant_kedge_ids:
                kedge = message["knowledge_graph"]["edges"][kedge_id]
                for attribute in kedge["attributes"]:
                    if attribute["attribute_type_id"] == "biolink:support_graphs":
                        auxgraph_ids.extend(attribute["value"])
            #for every supporting graph, get the edges
            all_relevant_edge_ids = set()
            for auxgraph_id in auxgraph_ids:
                try:
                    all_relevant_edge_ids.update(message["auxiliary_graphs"][auxgraph_id]["edges"])
                except KeyError:
                    #It looks like there are some upstream errors leading to auxgraph_ids that don't exist
                    logger.warning(f"Auxgraph id not found: {auxgraph_id}")
                    pass
            for edge_id in all_relevant_edge_ids:
                try:
                    edge = message["knowledge_graph"]["edges"][edge_id]
                except KeyError:
                    #this can only happen if the trapi is malformed, but we don't want to die if it is.
                    continue
                new_nonset_nodes.add(edge["subject"])
                new_nonset_nodes.add(edge["object"])
            new_nonset_nodes.update(nonset_nodes)
            lookup_nodes = list(new_nonset_nodes)

            # remove nodes that are not in node_pub_counts
            lookup_nodes = [n for n in lookup_nodes if n in node_pub_counts]
            lookup_nodes = sorted(lookup_nodes)
            for node_pair in combinations(lookup_nodes, 2):
                pair_to_answer[node_pair].add((ans_idx, analysis_idx))

            # For all nodes that are within sets, connect them to all nodes that are not in sets
            for qg_id, snodes in setnodes.items():
                for snode in snodes:
                    for node in lookup_nodes:
                        node_pair = tuple(sorted((node, snode)))
                        pair_to_answer[node_pair].add((ans_idx,analysis_idx))

            # now all nodes in set a to all nodes in set b
            for qga, qgb in combinations(setnodes.keys(), 2):
                for anode in setnodes[qga]:
                    for bnode in setnodes[qgb]:
                        # node_pair = tuple(sorted(anode, bnode))
                        node_pair = tuple(sorted((anode, bnode)))
                        pair_to_answer[node_pair].add((ans_idx, analysis_idx))
    return pair_to_answer
