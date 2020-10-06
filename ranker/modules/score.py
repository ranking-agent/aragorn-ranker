"""Rank."""
from reasoner_pydantic import Request, Message

from ranker.shared.util import flatten_semilist
from ranker.shared.ranker_obj import Ranker


async def query(request: Request, *, jaccard_like: bool = False) -> Message:
    """Score answers.

    This is mostly glue around the heavy lifting in ranker_obj.Ranker
    """
    message = request.message.dict()
    kgraph = message['knowledge_graph']
    answers = message['results']

    # resistance distance ranking
    pr = Ranker(message)
    answers = pr.rank(answers, jaccard_like=jaccard_like)

    # finish
    message['results'] = answers
    return Message(**message)
