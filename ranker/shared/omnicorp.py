"""Omnicorp support module."""
import logging
from .omnicorp_postgres import OmniCorp

logger = logging.getLogger(__name__)

COUNT_KEY = 'omnicorp_article_count'


class OmnicorpSupport():
    """Omnicorp support object."""

    def __init__(self):
        """Create omnicorp support object."""
        self.omnicorp = OmniCorp()

    async def __aenter__(self):
        """Enter context."""
        await self.omnicorp.connect()
        return self

    async def __aexit__(self, exception_type, exception_value, traceback):
        """Exit context, closing database connection."""
        await self.omnicorp.close()

    def term_to_term_pmids(self, node_a, node_b):
        """Get number of articles related to both terms and return the result."""
        articles = self.omnicorp.get_shared_pmids(node_a, node_b)
        return articles

    async def term_to_term_pmid_count(self, node_a, node_b):
        """Get number of articles related to both terms and return the result."""
        num_articles = await self.omnicorp.get_shared_pmids_count(node_a, node_b)
        return num_articles

    def node_pmids(self, node):
        """Get node publications."""
        pmids = self.omnicorp.get_pmids(node)
        return pmids

    async def node_pmid_count(self, node):
        """Get node publication count."""
        count = await self.omnicorp.count_pmids(node)
        return {COUNT_KEY: count}
