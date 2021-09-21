"""aragorn ranker server."""
import logging.config
import os
import pkg_resources
import yaml

from functools import wraps
from importlib import import_module
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from reasoner_pydantic import Response as PDResponse


APP = FastAPI(title='ARAGORN Ranker', version='2.0.3')

profiler = os.environ.get('PROFILER', False)
if profiler:
    from .profiler import profiler_middleware

# Set up default logger.
with pkg_resources.resource_stream('ranker', 'logging.yml') as f:
    config = yaml.safe_load(f.read())

logdir = 'logs'

if not os.path.exists(logdir):
    os.makedirs(logdir)

config['handlers']['file']['filename'] = os.path.join(logdir, 'ranker.log')

logging.config.dictConfig(config)

LOGGER = logging.getLogger(__name__)

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

dirname = os.path.join(os.path.dirname(__file__), 'modules')

operations = [
    op[:-3]
    for op in os.listdir(dirname)
    if op.endswith('.py') and not op.startswith('_')
]

for operation in operations:
    md = import_module(f"ranker.modules.{operation}")

    APP.post('/' + operation, tags=["ARAGORN-Ranker"], response_model=PDResponse, response_model_exclude_none=True, status_code=200)(log_exception(md.query))  # , response_model_exclude_unset=True


def construct_open_api_schema():

    if APP.openapi_schema:
        return APP.openapi_schema

    open_api_schema = get_openapi(
        title='ARAGORN Ranker',
        version='2.0.3',
        routes=APP.routes
    )

    open_api_extended_file_path = os.path.join(os.path.dirname(__file__), '../openapi-config.yaml')

    with open(open_api_extended_file_path) as open_api_file:
        open_api_extended_spec = yaml.load(open_api_file, Loader=yaml.SafeLoader)

    x_translator_extension = open_api_extended_spec.get("x-translator")
    x_trapi_extension = open_api_extended_spec.get("x-trapi")
    contact_config = open_api_extended_spec.get("contact")
    terms_of_service = open_api_extended_spec.get("termsOfService")
    servers_conf = open_api_extended_spec.get("servers")
    tags = open_api_extended_spec.get("tags")
    title_override = open_api_extended_spec.get("title") or 'ARAGORN Ranker'
    description = open_api_extended_spec.get("description")

    if tags:
        open_api_schema['tags'] = tags

    if x_translator_extension:
        # if x_translator_team is defined amends schema with x_translator extension
        open_api_schema["info"]["x-translator"] = x_translator_extension

    if x_trapi_extension:
        # if x_trapi_team is defined amends schema with x_trapi extension
        open_api_schema["info"]["x-trapi"] = x_trapi_extension

    if contact_config:
        open_api_schema["info"]["contact"] = contact_config

    if terms_of_service:
        open_api_schema["info"]["termsOfService"] = terms_of_service

    if description:
        open_api_schema["info"]["description"] = description

    if title_override:
        open_api_schema["info"]["title"] = title_override

    if servers_conf:
        for s in servers_conf:
            if s['description'].startswith('Default'):
                s['url'] = s['url'] + '/1.2'

        open_api_schema["servers"] = servers_conf

    return open_api_schema

# note: this must be commented out for local debugging
APP.openapi_schema = construct_open_api_schema()

APP.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
