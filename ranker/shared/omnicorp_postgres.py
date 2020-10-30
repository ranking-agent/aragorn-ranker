"""Omnicorp service module."""
import datetime
import os
import logging
import asyncpg
from ranker.shared.util import get_curie_prefix

logger = logging.getLogger(__name__)


OMNICORP_DB = os.environ.get('OMNICORP_DB', '')
OMNICORP_USER = os.environ.get('OMNICORP_USER', '')
OMNICORP_PORT = os.environ.get('OMNICORP_PORT', '')
OMNICORP_HOST = os.environ.get('OMNICORP_HOST', '')
OMNICORP_PASSWORD = os.environ.get('OMNICORP_PASSWORD', '')


class OmniCorp():
    """Omnicorp service object."""

    def __init__(self):
        """Create and omnicorp service object."""
        self.prefixes = set([
            'UBERON',
            'BSPO',
            'PATO',
            'GO',
            'MONDO',
            'HP',
            'ENVO',
            'OBI',
            'CL',
            'SO',
            'CHEBI',
            'HGNC',
            'EFO',
            'MESH'])
        self.pool = None
        self.nsingle = 0
        self.total_single_call = datetime.timedelta()
        self.npair = 0
        self.total_pair_call = datetime.timedelta()

    async def connect(self):
        """Connect to PostgreSQL."""
        logger.debug("Creating PostgreSQL connection pool...")
        self.pool = await asyncpg.create_pool(
            user=OMNICORP_USER,
            password=OMNICORP_PASSWORD,
            database=OMNICORP_DB,
            host=OMNICORP_HOST,
            port=OMNICORP_PORT,
        )

    async def close(self):
        """Close PostgreSQL connection."""
        logger.debug('Closing PostgreSQL connection pool...')
        await self.pool.close()

    async def get_shared_pmids_count(self, node1, node2):
        """Get shared PMIDs."""
        prefix1 = get_curie_prefix(node1)
        prefix2 = get_curie_prefix(node2)
        if (
                prefix1 not in self.prefixes or
                prefix2 not in self.prefixes
        ):
            return 0
        statement = (
            "SELECT COUNT(a.pubmedid)\n"
            f"FROM omnicorp.{prefix1} a\n"
            f"JOIN omnicorp.{prefix2} b ON a.pubmedid = b.pubmedid\n"
            "WHERE a.curie = $1\n"
            "AND b.curie = $2"
        )
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(statement, node1, node2)
        pmid_count = row['count']
        if pmid_count is None:
            logger.error("OmniCorp gave up")
            return None
        return pmid_count

    async def count_pmids(self, node):
        """Count PMIDs and return result."""
        if get_curie_prefix(node) not in self.prefixes:
            return 0
        prefix = get_curie_prefix(node)
        start = datetime.datetime.now()
        statement = (
            f"SELECT COUNT(pubmedid) from omnicorp.{prefix}\n"
            "WHERE curie = $1"
        )
        async with self.pool.acquire() as conn:
            row = await conn.fetch(statement, node)
        n = row[0]['count']
        end = datetime.datetime.now()
        self.total_single_call += (end - start)
        # logger.debug(f"""Found {n} pmids in {end-start}
        #             Total {self.total_single_call}""")
        self.nsingle += 1
        if self.nsingle % 100 == 0:
            logger.info(f"NCalls: {self.nsingle}\n" +
                        f"Total time: {self.total_single_call}\n" +
                        f"Avg Time: {self.total_single_call/self.nsingle}")
        return n
