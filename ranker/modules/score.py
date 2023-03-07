"""Rank."""
import logging
from fastapi.responses import JSONResponse
from reasoner_pydantic import Response as PDResponse
from ranker.shared.ranker_obj import Ranker
from ranker.shared.util import create_log_entry
import os
from datetime import datetime

logger = logging.getLogger(__name__)


async def query(request: PDResponse, *, jaccard_like: bool = False):
    """Score answers.

    This is mostly glue around the heavy lifting in ranker_obj.Ranker
    """
    # get the debug environment variable
    debug = os.environ.get("DEBUG_TIMING", "False")

    if debug == "True":
        dt_start = datetime.now()

    # get the message into a dict
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

    # get a reference to the entire message
    message = in_message["message"]

    message = in_message["message"]
    if ("results" not in message) or (message["results"] is None):
        #No results to weight. abort
        return JSONResponse(content=in_message, status_code=status_code)

    # get a reference to the results
    answers = message["results"]

    # get profile
    profile = in_message.get("profile")

    try:
        # resistance distance ranking
        pr = Ranker(message)

        # rank the answers. there should be a score for each bound result after this
        answers = pr.rank(answers, jaccard_like=jaccard_like)

        # save the results
        message["results"] = answers
    except Exception as e:
        # put the error in the response
        status_code = 500
        logger.exception(f"Aragorn-ranker/score exception {e}")
        # save any log entries
        # in_message['logs'].append(create_log_entry(f'Exception: {str(e)}', 'ERROR'))

    if debug == "True":
        diff = datetime.now() - dt_start
        in_message["logs"].append(
            create_log_entry(
                f"End of score processing. Time elapsed: {diff.seconds} seconds",
                "DEBUG",
            )
        )

    # return the result to the caller
    return JSONResponse(content=in_message, status_code=status_code)
