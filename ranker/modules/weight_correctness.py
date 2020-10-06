"""Weight edges."""
from collections import defaultdict
import math
from typing import Optional

from fastapi import Query
from reasoner_pydantic import Request, Message


async def query(
        request: Request,
        relevance: Optional[float] = Query(
            0.0025,
            description='portion of cooccurrence pubs relevant to question',
        ),
        wt_min: Optional[float] = Query(
            0.0,
            description='minimum weight (at 0 pubs)',
        ),
        wt_max: Optional[float] = Query(
            1.0,
            description='maximum weight (at inf pubs)',
        ),
        p50: Optional[float] = Query(
            2.0,
            description='pubs at 50% of wt_max',
        ),
) -> Message:
    """Weight kgraph edges based on metadata.

    "19 pubs from CTD is a 1, and 2 should at least be 0.5"
        - cbizon
    """
    message = request.message.dict()

    def sigmoid(x):
        """Scale with partial sigmoid - the right (concave down) half.

        Such that:
           f(0) = wt_min
           f(inf) = wt_max
           f(p50) = 0.5 * wt_max
        """
        a = 2 * (wt_max - wt_min)
        r = 0.5 * wt_max
        c = wt_max - 2 * wt_min
        k = 1 / p50 * (math.log(r + c) - math.log(a - r - c))
        return a / (1 + math.exp(-k * x)) - c

    kgraph = message['knowledge_graph']
    node_pubs = {n['id']: n.get('omnicorp_article_count', None) for n in kgraph['nodes']}
    all_pubs = 27840000

    results = message['results']

    # ensure that each edge_binding has a single kg_id
    for result in results:
        result['edge_bindings'] = [
            eb
            for ebs in result['edge_bindings']
            for eb in (
                [
                    {
                        'qg_id': ebs['qg_id'],
                        'kg_id': kg_id,
                    }
                    for kg_id in ebs['kg_id']
                ] if isinstance(ebs['kg_id'], list)
                else [ebs]
            )
        ]

    # map kedges to edge_bindings
    krmap = defaultdict(list)
    for result in results:
        for eb in result['edge_bindings']:
            assert isinstance(eb['kg_id'], str)
            eb['weight'] = eb.get('weight', 1.0)
            krmap[eb['kg_id']].append(eb)

    edges = kgraph['edges']
    for edge in edges:
        edge_pubs = edge.get('num_publications', len(edge.get('publications', [])))
        if edge['type'] == 'literature_co-occurrence':
            source_pubs = int(node_pubs[edge['source_id']])
            target_pubs = int(node_pubs[edge['target_id']])

            cov = (edge_pubs / all_pubs) - (source_pubs / all_pubs) * (target_pubs / all_pubs)
            cov = max((cov, 0.0))
            effective_pubs = cov * all_pubs * relevance
        else:
            effective_pubs = edge_pubs + 1  # consider the curation a pub

        for redge in krmap[edge['id']]:
            redge['weight'] = redge.get('weight', 1.0) * sigmoid(effective_pubs)

    message['knowledge_graph'] = kgraph
    return Message(**message)
