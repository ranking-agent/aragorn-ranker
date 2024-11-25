"""Reasoner filters utilities."""
import random
import string
from datetime import datetime
from typing import List


def batches(arr, n):
    """Iterate over arr by batches of size n."""
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def create_log_entry(msg: str, err_level, code=None) -> dict:
    """Creates a log entry"""
    now = datetime.now()

    # load the data
    ret_val = {
        'timestamp': now.strftime("%m-%d-%Y %H:%M:%S"),
        'level': err_level,
        'message': msg,
        'code': code
    }

    # return to the caller
    return ret_val
