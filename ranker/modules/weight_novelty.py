"""Novelty."""
import logging
import os
import json
import uuid
import redis
import itertools

from collections import defaultdict
from reasoner_pydantic import Response
from ranker.shared.qgraph_compiler import NodeReference, EdgeReference
from ranker.shared.util import batches
from ranker.shared.neo4j_ import Neo4jDatabase

logger = logging.getLogger(__name__)

AC_CACHE_HOST = os.environ.get('AC_CACHE_HOST', 'localhost')
AC_CACHE_PORT = os.environ.get('AC_CACHE_PORT', '1234')

NEO4J_URL = os.environ.get('NEO4J_URL', 'http://localhost:7474')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'pword')


def create_nodes_to_links(allnodes):
    p = get_redis_pipeline(0)
    # Create a dict from node->links by looking up in redis.  Each link is a potential node to add.
    # This is across all opportunities as well
    # Could pipeline
    nodes_to_links = {}
    for group in grouper(1000, allnodes.keys()):
        for node in group:
            p.get(node)
        linkstrings = p.execute()
        for node, linkstring in zip(group, linkstrings):
            if linkstring is None:
                print(node)
            if linkstring is None:
                links = []
            else:
                links = json.loads(linkstring)
            nodes_to_links[node] = links
    return nodes_to_links


def get_rgraph(result, message):
    """Get "ranker" subgraph."""
    qedges_by_id = {
        qedge['id']: qedge
        for qedge in message['query_graph']['edges']
    }
    kedges_by_id = {
        kedge['id']: kedge
        for kedge in message['knowledge_graph']['edges']
    }

    rnodes = result['node_bindings']

    # get "result" edges
    redges = []
    for eb in result['edge_bindings']:
        qedge_id = eb['qg_id']
        kedge_id = eb['kg_id']
        if qedge_id.startswith('s'):
            continue

        qedge = qedges_by_id[qedge_id]
        kedge = kedges_by_id[kedge_id]

        # find source and target
        # qedge direction may not match kedge direction
        # we'll go with the qedge direction
        kg_source_id = next(
            rnode['kg_id']
            for rnode in rnodes
            if (
                rnode['qg_id'] == qedge['source_id'] and
                (
                    kedge['source_id'] == rnode['kg_id'] or
                    kedge['target_id'] == rnode['kg_id']
                )
            )
        )
        kg_target_id = next(
            rnode['kg_id']
            for rnode in rnodes
            if (
                rnode['qg_id'] == qedge['target_id'] and
                (
                    kedge['source_id'] == rnode['kg_id'] or
                    kedge['target_id'] == rnode['kg_id']
                )
            )
        )
        edge = {
            'id': str(uuid.uuid4()),
            'eb': eb,
            'qg_source_id': qedge['source_id'],
            'qg_target_id': qedge['target_id'],
            'kg_source_id': kg_source_id,
            'kg_target_id': kg_target_id,
        }
        redges.append(edge)

    return {
        'nodes': rnodes,
        'edges': redges
    }


def get_redis_pipeline(dbnum):
    typeredis = redis.Redis(host=AC_CACHE_HOST, port=int(AC_CACHE_PORT), db=dbnum)

    p = typeredis.pipeline()

    return p


def get_link_counts(unique_links):
    # Now we are going to hit redis to get the counts for all of the links.
    # our unique_links are the keys
    p = get_redis_pipeline(2)
    lcounts = {}
    for ulg in grouper(1000, unique_links):
        for ul in ulg:
            p.get(str(ul))
        ns = p.execute()
        for ul, n in zip(ulg, ns):
            try:
                lcounts[ul] = int(n)
            except Exception as e:
                print('Failure')
                print(ul)
                raise e
    return lcounts


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


def uniquify_links(nodes_to_links, opportunities):
    # A link might occur for multiple nodes and across different opportunities
    # Create the total unique set of links
    unique_links = set()
    unique_link_nodes = set()
    io = 0
    for opportunity in opportunities:
        # print('new op', io, dt.now())
        io += 1
        kn = opportunity.get_kg_ids()
        # print('',len(kn))
        seen = set()
        for n in kn:
            # if len(nodes_to_links[n]) > 10000:
            #    print(' ',n,len(nodes_to_links[n]),opportunity.get_qg_semantic_type())
            for link in nodes_to_links[n]:
                # The link as defined uses the input node as is_source, but the lookup into redis uses the
                # linked node as the is_source, so gotta flip it
                lplus = link + [opportunity.get_qg_semantic_type()]
                lplus[2] = not lplus[2]
                tl = tuple(lplus)
                if tl in seen:
                    # if lplus[0] == 'NCBIGene:355':
                    #    print(io, tl)
                    unique_links.add(tl)
                    unique_link_nodes.add(tl[0])
                else:
                    seen.add(tl)
        # print('','end',dt.now())
    return unique_link_nodes, unique_links


async def query(response: Response, *, exclude_sets=False) -> Response:
    """Compute informativeness weights for edges."""
    message = response.message.dict()
    qgraph = message['query_graph']
    results = message['results']

    qnodes = qgraph['nodes']
    qedges = qgraph['edges']

    # knode_map = {knode['id']: knode for knode in knodes}
    qnode_map = {qnode['id']: qnode for qnode in qnodes}
    qedge_map = {qedge['id']: qedge for qedge in qedges}

    driver = Neo4jDatabase(
        url=NEO4J_URL,
        credentials={
            'username': NEO4J_USER,
            'password': NEO4J_PASSWORD,
        },
    )
    redges_by_id = dict()
    count_plans = defaultdict(lambda: defaultdict(list))
    for kdx, result in enumerate(results):
        rgraph = get_rgraph(result, message)
        redges_by_id.update({
            (kdx, redge['id']): redge
            for redge in rgraph['edges']
        })

        for redge in rgraph['edges']:
            if (not exclude_sets) or qnode_map[redge['qg_target_id']].get('set', False):
                count_plans[redge['kg_source_id']][(redge['eb']['qg_id'], redge['qg_target_id'])].append(
                    (kdx, redge['id'])
                )
            if (not exclude_sets) or qnode_map[redge['qg_source_id']].get('set', False):
                count_plans[redge['kg_target_id']][(redge['eb']['qg_id'], redge['qg_source_id'])].append(
                    (kdx, redge['id'])
                )

    count_to_redge = {}
    for ldx, batch in enumerate(batches(list(count_plans.keys()), 1000)):
        batch_bits = []
        for idx, ksource_id in enumerate(batch):
            sets = []
            plan = count_plans[ksource_id]
            anchor_node_reference = NodeReference({
                'id': f'n{idx:04d}',
                'curie': ksource_id,
                'type': 'named_thing'
            })
            anchor_node_reference = str(anchor_node_reference)

            # base = f"MATCH ({anchor_node_reference}) "

            for jdx, (qlink, redge_ids) in enumerate(plan.items()):
                cypher_counts = []
                qedge_id, qtarget_id = qlink
                count_id = f"c{idx:03d}{chr(97 + jdx)}"
                qedge = qedge_map[qedge_id]
                edge_reference = EdgeReference(qedge, anonymous=True)
                anon_node_reference = NodeReference({
                    **qnode_map[qtarget_id],
                    'id': count_id,
                })

                if qedge['source_id'] == qtarget_id:
                    source_reference = anon_node_reference
                    target_reference = anchor_node_reference
                elif qedge['target_id'] == qtarget_id:
                    source_reference = anchor_node_reference
                    target_reference = anon_node_reference

                # TODO: source_reference and target_reference referenced before assignent

                cypher_counts.append(f"{anon_node_reference.name}: count(DISTINCT {anon_node_reference.name})")
                count_to_redge[count_id] = redge_ids
                sets.append(f'MATCH ({source_reference}){edge_reference}({target_reference})' + ' RETURN {' + ', '.join(cypher_counts) + '} as output')
            batch_bits.append(' UNION ALL '.join(sets))
        cypher = ' UNION ALL '.join(batch_bits)
        response = driver.run(cypher)

        degrees = {
            key: value
            for result in response
            for key, value in result['output'].items()
        }

        for key in degrees:
            for redge_id in count_to_redge[key]:
                eb = redges_by_id[redge_id]['eb']
                eb['weight'] = eb.get('weight', 1.0) / degrees[key]

    message['results'] = results

    # get this in the correct response model format
    ret_val = {'message': message}

    # return the message back to the caller
    return Response(**ret_val)
