"""Literature co-occurrence support."""
import logging
import os
import asyncio
from uuid import uuid4
from collections import defaultdict
from itertools import combinations
from reasoner_pydantic import Response, Message
from ranker.shared.cache import Cache
from ranker.shared.omnicorp import OmnicorpSupport
from ranker.shared.util import batches

logger = logging.getLogger(__name__)

CACHE_HOST = os.environ.get('CACHE_HOST', 'localhost')
CACHE_PORT = os.environ.get('CACHE_PORT', '6379')
CACHE_DB = os.environ.get('CACHE_DB', '0')
CACHE_PASSWORD = os.environ.get('CACHE_PASSWORD', '')


async def count_node_pmids(supporter, node, key, value, cache, kgraph):
    """Count node PMIDs and add as node property."""
    if value is not None:
        logger.debug(f'{key} is cached')
        support_dict = value
    else:
        logger.debug(f'Computing {key}...')
        support_dict = await supporter.node_pmid_count(kgraph[node])
        if cache and support_dict['omnicorp_article_count']:
            cache.set(key, support_dict)
    # add omnicorp_article_count to nodes in networkx graph
    attribute = [{'type': 'omnicorp_article_count', 'value': support_dict['omnicorp_article_count']}]
    kgraph[node]['attributes'] = attribute
    #kgraph[node].update(support_dict)


async def count_shared_pmids(
        supporter, support_idx, pair, key, value,
        cache, cached_prefixes, kgraph, pair_to_answer,
        answers,
):
    """Count PMIDS shared by a pair of nodes and create a new support edge."""
    support_edge = value

    if support_edge is None:
        # There are two reasons that we don't get anything back:
        # 1. We haven't evaluated that pair
        # 2. We evaluated, and found it to be zero, and it was part
        #    of a prefix pair that we evaluated all of.  In that case
        #    we can infer that getting nothing back means an empty list
        #    check cached_prefixes for this...
        prefixes = tuple(ident.split(':')[0].upper() for ident in pair)
        if cached_prefixes and prefixes in cached_prefixes:
            logger.debug(f'{pair} should be cached: assume 0')
            support_edge = []
        else:
            logger.debug(f'Computing {pair}...')
            support_edge = await supporter.term_to_term_pmid_count(pair[0], pair[1])
            if cache and support_edge:
                cache.set(key, support_edge)
    else:
        logger.debug(f'{pair} is cached')
    if not support_edge:
        return

    uid = str(uuid4())

    kgraph['edges'].update({uid: {
        'predicate': 'biolink:literature_co_occurrence',
        'attributes': [
            {'type': 'num_publications', 'value': support_edge},
            {'type': 'publications', 'value': []},
            {'type': 'source_database', 'value': 'omnicorp'},
            {'type': 'edge_source', 'value': 'omnicorp.term_to_term'}
        ],
        'subject': pair[0],
        'object': pair[1],
    }})

    for sg in pair_to_answer[pair]:
        answers[sg]['edge_bindings'].update({f's{support_idx}': [{'id': uid}]})


async def query(response: Response):
    """Add support to message.

    Add support edges to knowledge_graph and bindings to results.
    """
    message = response.message.dict()

    qgraph = message['query_graph']
    kgraph = message['knowledge_graph']
    answers = message['results']

    # get cache if possible
    try:
        cache = Cache(
            redis_host=CACHE_HOST,
            redis_port=CACHE_PORT,
            redis_db=CACHE_DB,
            redis_password=CACHE_PASSWORD,
        )
    except Exception as err:
        logger.exception(err)
        cache = None

    redis_batch_size = 100

    async with OmnicorpSupport() as supporter:
        # get all node supports

        keys = [f"{supporter.__class__.__name__}({node})" for node in kgraph['nodes']]
        values = []
        for batch in batches(keys, redis_batch_size):
            values.extend(cache.mget(*batch))

        jobs = [
            count_node_pmids(supporter, node, key, value, cache, kgraph['nodes'])
            for node, value, key in zip(kgraph['nodes'], values, keys)
        ]

        # which qgraph nodes are sets?
        qgraph_setnodes = set([n for n in qgraph['nodes'] if (('is_set' in qgraph['nodes'][n]) and (qgraph['nodes'][n]['is_set']))])

        # Generate a set of pairs of node curies
        pair_to_answer = defaultdict(set)  # a map of node pairs to answers
        for ans_idx, answer_map in enumerate(answers):

            # Get all nodes that are not part of sets and densely connect them
            # can be str (not a set) or list (could be a set or not a set)
            nonset_nodes = []
            setnodes = {}
            # Note: modified by powen for trapi 1.0.
            # node binding results is now a dict containing dicts that contain a list of dicts.
            for nb in answer_map['node_bindings']:
                if nb in qgraph_setnodes:
                    setnodes[nb] = answer_map['node_bindings'][nb][0]['id']
                else:
                    nonset_nodes.append(answer_map['node_bindings'][nb][0]['id'])

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
                        node_pair = tuple(sorted(anode, bnode))
                        pair_to_answer[node_pair].add(ans_idx)

        # get all pair supports
        cached_prefixes = cache.get('OmnicorpPrefixes') if cache else None

        keys = [f"{supporter.__class__.__name__}_count({pair[0]},{pair[1]})" for pair in pair_to_answer]
        values = []
        for batch in batches(keys, redis_batch_size):
            values.extend(cache.mget(*batch))

        jobs.extend([
            count_shared_pmids(
                supporter, support_idx, pair, key, value,
                cache, cached_prefixes, kgraph, pair_to_answer,
                answers,
            )
            for support_idx, (pair, value, key) in enumerate(zip(pair_to_answer, values, keys))
        ])
        await asyncio.gather(*jobs)

    # load the new results into the response
    message['knowledge_graph'] = kgraph
    message['results'] = answers

    # get this in the correct response model format
    ret_val = {'message': message}

    # return the results
    return Response(**ret_val)
