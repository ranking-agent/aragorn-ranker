"""Rank."""
from reasoner_pydantic import Response, Message

from ranker.shared.util import flatten_semilist
from ranker.shared.ranker_obj import Ranker


async def query(response: Response, *, jaccard_like: bool = False) -> Response:
    """Score answers.

    This is mostly glue around the heavy lifting in ranker_obj.Ranker
    """
    message = response.message.dict()
    kgraph = message['knowledge_graph']
    answers = message['results']

    # resistance distance ranking
    pr = Ranker(message)
    answers = pr.rank(answers, jaccard_like=jaccard_like)

    # finish
    message['results'] = answers

    # get this in the correct response model format
    ret_val = {'message': message}

    # return the results
    return Response(**ret_val)
