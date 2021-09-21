"""Rank."""
from fastapi.responses import JSONResponse
from reasoner_pydantic import Response as PDResponse
from ranker.shared.ranker_obj import Ranker
from ranker.shared.util import create_log_entry
import os
from datetime import datetime


async def query(request: PDResponse, *, jaccard_like: bool = False):
    """Score answers.

    This is mostly glue around the heavy lifting in ranker_obj.Ranker
    """
    # get the debug environment variable
    debug = os.environ.get('SC_DEBUG', False)

    if debug:
        dt_start = datetime.now()

    # get the message into a dict
    in_message = request.dict()

    # save the logs for the response (if any)
    if 'logs' not in in_message or in_message['logs'] is None:
        in_message['logs'] = []

    # init the status code
    status_code: int = 200

    # get a reference to the entire message
    message = in_message['message']

    # get a reference to the results
    answers = message['results']

    try:
        # resistance distance ranking
        pr = Ranker(message)

        # rank the answers. there should be a score for each bound result after this
        answers = pr.rank(answers, jaccard_like=jaccard_like)

        # save the results
        message['results'] = answers
    except Exception as e:
        # put the error in the response
        status_code = 500

        # save any log entries
        in_message['logs'].append(create_log_entry(f'Exception: {str(e)}', 'ERROR'))

    if debug:
        diff = datetime.now() - dt_start
        in_message['logs'].append(create_log_entry(f'End of score processing. Time elapsed: {diff.seconds} seconds', 'DEBUG'))

    # return the result to the caller
    return JSONResponse(content=in_message, status_code=status_code)