"""Literature co-occurrence support."""
import logging
import os
import asyncio

from uuid import uuid4
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from fastapi.responses import JSONResponse
from ranker.shared.cache import Cache
from ranker.shared.omnicorp import OmnicorpSupport
from ranker.shared.util import batches, create_log_entry
from reasoner_pydantic import Response as PDResponse

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

        for sg in pair_to_answer[pair]:
            answers[sg]["edge_bindings"].update({f"s{support_idx}": [{"id": uid}]})


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
        # get all node supports

        keys = list(kgraph["nodes"].keys())

        values = {}
        for batch in batches(keys, redis_batch_size):
            values.update(cache.mquery(batch,"OMNICORP",
                                       "as x MATCH (q:CURIE {concept:x}) return q.concept, q.publication_count"))

        await add_node_pmid_counts(kgraph,values)

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

        pair_to_answer = await generate_curie_pairs(answers, qgraph_setnodes)

        keys = [ list(x) for x in pair_to_answer.keys() ]
        values = {}
        for batch in batches(keys, redis_batch_size):
            values.update(cache.mquery(batch,"OMNICORP",
                                       "as q MATCH (a:CURIE {concept:q[0]})-[x]-(b:CURIE {concept:q[1]}) return q[0],q[1],x.publication_count"))

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


async def generate_curie_pairs(answers, qgraph_setnodes):
    # Generate a set of pairs of node curies
    pair_to_answer = defaultdict(set)  # a map of node pairs to answers
    for ans_idx, answer_map in enumerate(answers):

        # Get all nodes that are not part of sets and densely connect them
        # can be str (not a set) or list (could be a set or not a set)
        nonset_nodes = []
        setnodes = {}

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

        nonset_nodes = sorted(nonset_nodes)
        # nodes = sorted([nb['kg_id'] for nb in answer_map['node_bindings'] if isinstance(nb['kg_id'], str)])
        for node_pair in combinations(nonset_nodes, 2):
            pair_to_answer[node_pair].add(ans_idx)

        # set_nodes_list_list = [nb['kg_id'] for nb in answer_map['node_bindings'] if isinstance(nb['kg_id'], list)]
        # set_nodes = [n for el in set_nodes_list_list for n in el]
        # For all nodes that are within sets, connect them to all nodes that are not in sets
        for qg_id, snodes in setnodes.items():
            for snode in snodes:
                for node in nonset_nodes:
                    node_pair = tuple(sorted((node, snode)))
                    pair_to_answer[node_pair].add(ans_idx)

        # now all nodes in set a to all nodes in set b
        for qga, qgb in combinations(setnodes.keys(), 2):
            for anode in setnodes[qga]:
                for bnode in setnodes[qgb]:
                    # node_pair = tuple(sorted(anode, bnode))
                    node_pair = tuple(sorted((anode, bnode)))
                    pair_to_answer[node_pair].add(ans_idx)
    return pair_to_answer
