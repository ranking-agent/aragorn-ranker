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
CACHE_PORT = os.environ.get("CACHE_PORT", "6379")
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
    kgraph,
    values,
    pair_to_answer,
    answers,
):
    """Count PMIDS shared by a pair of nodes and create a new support edge."""
    support_idx = 0
    for pair, publication_count in values.items():
        support_idx += 1
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
                        # If we're returning a count, then returning an empty list here is gibberish, and causes an error in weighting.
                        # {'original_attribute_name': 'publications', 'attribute_type_id': 'biolink:publications', 'value_type_id': 'EDAM:data_0006', 'value': []},
                        {
                            "attribute_type_id": "biolink:primary_knowledge_source",  # the ‘key’*
                            "value": "infores:omnicorp",
                            "value_type_id": "biolink:InformationResource",
                            "attribute_source": "infores:aragorn",
                        },
                    ],
                    "subject": pair[0],
                    "object": pair[1],
                }
            }
        )

        # pair_to_answer is a dictionary of pairs to tuples.
        # Each tuple is a pair of (answer_idx, analysis_idx)
        for answer_idx, analysis_idx in pair_to_answer[pair]:
            answers[answer_idx]["analyses"][analysis_idx]["edge_bindings"].update({f"s{support_idx}": [{"id": uid}]})


def create_node2pairs(pairs: Set[Tuple]) -> Dict[str, List[Tuple]] :
    """Create a dictionary of node to pairs."""
    node2pairs = defaultdict(list)
    for pair in pairs:
        for node in pair:
            node2pairs[node].append(pair)
    return node2pairs


async def query(request: PDResponse):
    """Add support to message.

    Add support edges to knowledge_graph and bindings to results.
    """
    # get the debug environment variable
    debug = os.environ.get("DEBUG_TIMING", "False")

    if debug == "True":
        dt_start = datetime.now()

    in_message = request.dict(exclude_none=True, exclude_unset=True)

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

    redis_batch_size = 100

    try:
        # get all node supports.

        # first, get the publication counts for all nodes in the kgraph
        keys = list(kgraph["nodes"].keys())

        node_pub_counts = {}
        for batch in batches(keys, redis_batch_size):
            node_pub_counts.update(cache.mquery(batch,"OMNICORP",
                                       "as x MATCH (q:CURIE {concept:x}) return q.concept, q.publication_count"))

        await add_node_pmid_counts(kgraph, node_pub_counts)

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
        #Note, this will be affected by TRAPI 1.4
        pair_to_answer = await generate_curie_pairs(answers, qgraph_setnodes, node_pub_counts, message)

        #Now, the simplest thing to do would be to go to redisgraph and look up each pair.   However, it turns out that
        # the query (a)-[x]-(b) with a and b specified is unreasonably slow.  It's slow because it is getting all the
        # nodes connected to a and all the nodes connected to b and then intersecting.  Frankly, it seems stupid.
        # Because the same node will be repeated in many pairs, not only are you doing a slow query, but you're doing
        # it over and over again.   So, we take a different approach

        # We are just going to do all the onehops from each node in any pair and then do everything else in python.
        # What that looks like is: first, remove any nodes that didn't have publications from above, because they
        # by definition won't have any lit co-occurrence edges.  Then find all the pairs and group them by nodes.
        # Starting with the node in the most pairs, query each node for all of its neighbors.  Collect all the relevant
        # counts, and remove the pairs that have been accounted for.  Repeat until all pairs have been accounted for.

        # The only other caveat concerns batching.   We actually query for the top N nodes at a time, which means
        # we might get the same pair coming back twice times in the same batch.  But it doesn't matter much.

        # the pairs are tuples (sorted)
        pairs = set(list( pair_to_answer.keys() ))

        values = {}
        #The batch size here is a bit tricky.   Every time we get back a batch of counts, we are removing pairs,
        # which also means that we may be removing nodes.   So if we did everything at once, we're checking
        # for a lot of nodes that we just don't need to.   But if we query on smaller batches to make sure that never
        # happens, then lots of queries.  But redisgraph is already pretty low latency, so maybe it doesn't hurt that
        # badly.  Also - it turns out that the one hop query is pretty slow, so the latency makes up a smaller
        # fraction of the total cost anyway.
        redis_batch_size = 10
        while len(pairs) > 0:
            node2pairs = create_node2pairs(pairs)
            top_node_items = heapq.nlargest(redis_batch_size, node2pairs.items(), key=lambda x: len(x[1]))
            top_nodes = [x[0] for x in top_node_items]
            all_counts = cache.mquery(top_nodes,"OMNICORP", "as q MATCH (a:CURIE {concept:q})-[x]-(b:CURIE) return q,b.concept,x.publication_count")
            #collect the counts
            for (node1, node2), count in all_counts.items():
                pair = tuple(sorted([node1, node2]))
                #We're getting everything linked to top node, whether it's in our answer or not
                if pair in pairs:
                    values[pair] = count
            # remove all the related pairs whether they come back in all_counts or not
            for node in top_nodes:
                for pair in node2pairs[node]:
                    # it's possible that the pair has already been removed
                    if pair in pairs:
                        pairs.remove(pair)

        await add_shared_pmid_counts(kgraph,values,pair_to_answer,answers)

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
                    nonset_nodes.append(
                        answer_map["node_bindings"][nb][0]["id"]
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
                all_relevant_edge_ids.update(message["auxiliary_graphs"][auxgraph_id]["edges"])
            for edge_id in all_relevant_edge_ids:
                edge = message["knowledge_graph"]["edges"][edge_id]
                new_nonset_nodes.add(edge["subject"])
                new_nonset_nodes.add(edge["object"])
            lookup_nodes = nonset_nodes + list(new_nonset_nodes)

            # remove nodes that are not in node_pub_counts
            lookup_nodes = [n for n in lookup_nodes if n in node_pub_counts]
            lookup_nodes = sorted(lookup_nodes)
            for node_pair in combinations(nonset_nodes, 2):
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
