"""Neo4j lookup utilities."""
from abc import ABC, abstractmethod
from copy import deepcopy
import json
import logging
from urllib.parse import urlparse

import httpx
from neo4j import GraphDatabase, basic_auth

from messenger.shared.util import batches, flatten_semilist

logger = logging.getLogger(__name__)


class Neo4jDatabase():
    """Neo4j database.

    This is a thin wrapper that chooses whether to instantiate
    a Bolt interface or an HTTP interface.
    """

    def __new__(cls, url=None, **kwargs):
        """Generate database interface."""
        scheme = urlparse(url).scheme
        if scheme == 'http':
            return HttpInterface(url=url, **kwargs)
        elif scheme == 'bolt':
            return BoltInterface(url=url, **kwargs)
        else:
            raise ValueError(f'Unsupported interface scheme "{scheme}"')

    @abstractmethod
    def run(self, statement, *args):
        """Run statement."""
        pass


class Neo4jInterface(ABC):
    """Abstract interface to Neo4j database."""

    def __init__(self, url=None, credentials=None, **kwargs):
        """Initialize."""
        url = urlparse(url)
        self.hostname = url.hostname
        self.port = url.port
        self.auth = (credentials['username'], credentials['password'])

    @abstractmethod
    def run(self, statement, *args):
        """Run statement."""
        pass


class HttpInterface(Neo4jInterface):
    """HTTP interface to Neo4j database."""

    def __init__(self, **kwargs):
        """Initialize."""
        super().__init__(**kwargs)
        self.url = f'http://{self.hostname}:{self.port}/db/data/transaction/commit'

    async def arun(self, statement, *args):
        """Run statement."""
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                self.url,
                auth=self.auth,
                json={"statements": [{"statement": statement}]},
            )
        result = response.json()['results'][0]
        result = [
            dict(zip(result['columns'], datum['row']))
            for datum in result['data']
        ]
        return result

    def run(self, statement, *args):
        """Run statement."""
        response = httpx.post(
            self.url,
            auth=self.auth,
            json={"statements": [{"statement": statement}]},
        )
        result = response.json()['results'][0]
        result = [
            dict(zip(result['columns'], datum['row']))
            for datum in result['data']
        ]
        return result


class BoltInterface(Neo4jInterface):
    """Bolt interface to Neo4j database."""

    def __init__(self, **kwargs):
        """Initialize."""
        super().__init__(**kwargs)
        self.url = f'bolt://{self.hostname}:{self.port}'
        self.driver = GraphDatabase.driver(
            self.url,
            auth=basic_auth(*self.auth)
        )

    def run(self, statement, *args):
        """Run statement."""
        with self.driver.session() as session:
            return [dict(row) for row in session.run(statement)]
