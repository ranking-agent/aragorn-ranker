"""aragorn ranker server."""
import logging.config
import os
import pkg_resources
import yaml

from functools import wraps
from importlib import import_module
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from reasoner_pydantic import Response

# Set up default logger.
with pkg_resources.resource_stream('ranker', 'logging.yml') as f:
    config = yaml.safe_load(f.read())
logdir = 'logs'
if not os.path.exists(logdir):
    os.makedirs(logdir)
config['handlers']['file']['filename'] = os.path.join(logdir, 'ranker.log')
logging.config.dictConfig(config)

LOGGER = logging.getLogger(__name__)

APP = FastAPI(
    title='ARAGORN Ranker',
    version='2.2.0',
)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dirname = os.path.join(os.path.dirname(__file__), 'modules')
operations = [
    op[:-3]
    for op in os.listdir(dirname)
    if op.endswith('.py') and not op.startswith('_')
]


def log_exception(method):
    """Wrap method."""
    @wraps(method)
    async def wrapper(*args, **kwargs):
        """Log exception encountered in method, then pass."""
        try:
            return await method(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.exception(err)
            raise
    return wrapper


for operation in operations:
    md = import_module(f"ranker.modules.{operation}")

    APP.post('/' + operation, response_model=Response, response_model_exclude_none=True)(log_exception(md.query))
