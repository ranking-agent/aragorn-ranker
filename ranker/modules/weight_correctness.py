"""Weight edges."""
import math
from collections import defaultdict
from typing import Optional
from fastapi import Query
from reasoner_pydantic import Response, Message


async def query(
        response: Response,
        relevance: Optional[float] = Query(
            0.0025,
            description='Portion of cooccurrence publications relevant to a question',
        ),
        wt_min: Optional[float] = Query(
            0.0,
            description='Minimum weight (at 0 publications)',
        ),
        wt_max: Optional[float] = Query(
            1.0,
            description='Maximum weight (at inf publications)',
        ),
        p50: Optional[float] = Query(
            2.0,
            description='Publications at 50% of wt_max',
        ),
) -> Response:
    """Weight kgraph edges based on metadata.

    "19 pubs from CTD is a 1, and 2 should at least be 0.5"
        - cbizon
    """
    message = response.message.dict()

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

    # constant count of all publications
    all_pubs = 27840000

    # get the data nodes we need
    results = message['results']
    kgraph = message['knowledge_graph']

    # storage for the publication counts for the node
    node_pubs: dict = {}

    # for each node in the knowledge graph
    for n in kgraph['nodes']:
        # init the count value
        omnicorp_article_count: int = 0

        # get the article count atribute
        for p in kgraph['nodes'][n]['attributes']:
            # is this what we are looking for
            if p['name'] == 'omnicorp_article_count':
                # save it
                omnicorp_article_count = p['value']

                # no need to continue
                break

        # add the node d and count to the dict
        node_pubs.update({n: omnicorp_article_count})

    # map kedges to edge_bindings
    krmap = defaultdict(list)

    # for each reult listd in the data
    for result in results:
        # for every edge binding result
        for eb in result['edge_bindings']:
            # default the weight to 0 if there is none listed
            result['edge_bindings'][eb][0]['weight'] = result['edge_bindings'][eb][0].get('weight', 1)

            # get a reference to the weight for easy update later
            krmap[result['edge_bindings'][eb][0]['id']] = result['edge_bindings'][eb][0]

    # get the knowledge graph edges
    edges = kgraph['edges']

    # for each knowledge graph edge
    for edge in edges:
        # init theeffective publication count
        effective_pubs = 0

        # We are getting some results back (BTE?) that have "publications": ['PMID:1234|2345|83984']
        attributes = edges[edge].get('attributes', [])

        # init storage for the publications and their count
        publications = []
        num_publications = 0

        # for each data attribute
        for attribute in attributes:
            if attribute['name'] is not None:
                # is this the publication list
                if attribute['name'].startswith('publications'):
                    publications = attribute['value']
                # else is this the number of publications
                elif attribute['name'].startswith('num_publications'):
                    num_publications = attribute.get('value', 0)

        # if there was only 1 publication value found insure it wasnt a character seperated list
        if len(publications) == 1:
            if '|' in publications[0]:
                publications = publications[0].split('|')
            elif ',' in publications[0]:
                publications = publications[0].split(',')

            # get the real publication count
            num_publications = len(publications)

        # if there was no publication count found yet revert to the number of individual values
        if num_publications == 0:
            num_publications = len(publications)

        #now the nicer cleaner version when we have publications as an actual array
        #edge_pubs = edge.get('num_publications', len(edge.get('publications', [])))
        if edges[edge].get('predicate') == 'literature_co-occurrence':
            subject_pubs = int(node_pubs[edge['subject']])
            object_pubs = int(node_pubs[edge['object']])

            cov = (num_publications / all_pubs) - (subject_pubs / all_pubs) * (object_pubs / all_pubs)
            cov = max((cov, 0.0))
            effective_pubs = cov * all_pubs * relevance
        else:
            effective_pubs = num_publications + 1  # consider the curation a pub

        # save the weight value in the results using the reference shortcut above
        krmap[edge]['weight'] = krmap[edge].get('weight', 1.0) * sigmoid(effective_pubs)

    # save the new knowledge graph data
    message['knowledge_graph'] = kgraph

    # get this in the correct response model format
    ret_val = {'message': message}

    # return the message back to the caller
    return Response(**ret_val)
