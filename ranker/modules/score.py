"""Rank."""
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from reasoner_pydantic import Response as PDResponse
from ranker.shared.ranker_obj import Ranker


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


async def query(request: PDResponse, *, jaccard_like: bool = False):
    """Score answers.

    This is mostly glue around the heavy lifting in ranker_obj.Ranker
    """
    in_message = request.dict()

    # save the logs for the response (if any)
    if 'logs' not in in_message or in_message['logs'] is None:
        in_message['logs'] = []

    # init the status code
    status_code: int = 200

    message = in_message['message']

    kgraph = message['knowledge_graph']
    answers = message['results']

    try:
        # resistance distance ranking
        pr = Ranker(message)

        answers = pr.rank(answers, jaccard_like=jaccard_like)

        # finish
        message['results'] = answers

        # get this in the correct response model format
        ret_val = {'message': message}

    except Exception as e:
        # put the error in the response
        status_code = 500

        # save any log entries
        in_message['logs'].append(create_log_entry(f'Exception: {str(e)}', 'ERROR'))

    # validate the response again after normalization
    in_message = jsonable_encoder(PDResponse(**in_message))

    # return the result to the caller
    return JSONResponse(content=in_message, status_code=status_code)