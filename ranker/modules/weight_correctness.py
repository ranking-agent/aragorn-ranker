"""Weight edges."""
import math

from collections import defaultdict
from typing import Optional
from fastapi import Query
from datetime import datetime
from reasoner_pydantic import Response as PDResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


async def query(
        request: PDResponse,
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
):
    """Weight kgraph edges based on metadata.

    "19 pubs from CTD is a 1, and 2 should at least be 0.5"
        - cbizon
    """
    in_message = request.dict()


    # save the logs for the response (if any)
    if 'logs' not in in_message or in_message['logs'] is None:
        in_message['logs'] = []

    # init the status code
    status_code: int = 200

    message = in_message['message']

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

    def create_log_entry(msg: str, err_level, code=None) -> dict:
        # load the data
        ret_val = {
            'timestamp': str(datetime.now()),
            'level': err_level,
            'message': msg,
            'code': code
        }

        # return to the caller
        return ret_val

    try:
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
                if p['original_attribute_name'] == 'omnicorp_article_count':
                    # save it
                    omnicorp_article_count = p['value']

                    # no need to continue
                    break

            # add the node d and count to the dict
            node_pubs.update({n: omnicorp_article_count})

        # map kedges to result edge bindings
        krmap = defaultdict(list)

        # for each result listed in the data get a map reference and default the weight attribute
        for result in results:
            # for every edge binding result
            for eb in result['edge_bindings']:
                # loop through the edge binding
                for idx, binding_val in enumerate(result['edge_bindings'][eb]):
                    # get a reference to the weight for easy update later
                    krmap[binding_val['id']] = result['edge_bindings'][eb][idx]

                    found = False

                    # is there already a list of attributes
                    if 'attributes' in krmap[binding_val['id']] and krmap[binding_val['id']]['attributes'] is not None:
                        # loop through the attributes
                        for item in krmap[binding_val['id']]['attributes']:
                            # search for the weight attribute
                            if item['original_attribute_name'].startswith('weight'):
                                found = True
                                break

                    # was the attribute found
                    if not found:
                        if 'attributes' not in krmap[binding_val['id']] or krmap[binding_val['id']]['attributes'] is None:
                            krmap[binding_val['id']]['attributes'] = []

                        # create an Attribute
                        krmap[binding_val['id']]['attributes'].append({
                            'original_attribute_name': 'weight',
                            'attribute_type_id': 'biolink:has_numeric_value',
                            'value': 1,
                            'value_type_id': 'EDAM:data_1669'})

        # get the knowledge graph edges
        edges = kgraph['edges']

        # for each knowledge graph edge
        for edge in edges:
            # We are getting some results back (BTE?) that have "publications": ['PMID:1234|2345|83984']
            attributes = edges[edge].get('attributes', None)

            # init storage for the publications and their count
            publications = []
            num_publications = 0

            if attributes is not None:
                # for each data attribute collect the needed params
                for attribute in attributes:
                    if attribute['original_attribute_name'] is not None:
                        # is this the publication list
                        if attribute['original_attribute_name'].startswith('publications'):
                            publications = attribute['value']
                        # else is this the number of publications
                        elif attribute['original_attribute_name'].startswith('num_publications'):
                            num_publications = attribute.get('value', 0)

                # if there was only 1 publication value found insure it wasnt a character separated list
                if len(publications) == 1:
                    if '|' in publications[0]:
                        publications = publications[0].split('|')
                    elif ',' in publications[0]:
                        publications = publications[0].split(',')

                    # get the real publication count
                    num_publications = len(publications)

                # if there was no publication count found revert to the number of individual values
                if num_publications == 0:
                    num_publications = len(publications)

                # now the nicer cleaner version when we have publications as an actual array
                # edge_pubs = edge.get('num_publications', len(edge.get('publications', [])))
                if edges[edge].get('predicate') == 'literature_co-occurrence':
                    subject_pubs = int(node_pubs[edge['subject']])
                    object_pubs = int(node_pubs[edge['object']])

                    cov = (num_publications / all_pubs) - (subject_pubs / all_pubs) * (object_pubs / all_pubs)
                    cov = max((cov, 0.0))
                    effective_pubs = cov * all_pubs * relevance
                else:
                    effective_pubs = num_publications + 1  # consider the curation a pub

                # if there is something to add this new attribute to
                if len(krmap[edge]) != 0:
                    # is there already a list of attributes
                    if 'attributes' in krmap[edge]:
                        # loop through the attributes
                        for item in krmap[edge]['attributes']:
                            # search for the weight attribute
                            if item['original_attribute_name'].startswith('weight'):
                                # update the params
                                item['attribute_type_id'] = 'biolink:has_numeric_value'
                                item['value'] = item['value'] * sigmoid(effective_pubs)
                                item['value_type_id'] = 'EDAM:data_1669'
                                found = True
                                break

        # save the new knowledge graph data
        message['knowledge_graph'] = kgraph

    except Exception as e:
        # put the error in the response
        status_code = 500

        # save any log entries
        in_message['logs'].append(create_log_entry(f'Exception: {str(e)}', 'ERROR'))

    # validate the response again after normalization
    in_message = jsonable_encoder(PDResponse(**in_message))

    # return the result to the caller
    return JSONResponse(content=in_message, status_code=status_code)
