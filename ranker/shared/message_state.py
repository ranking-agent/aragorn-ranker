"""Message state analysis tools."""
import pkg_resources
import jsonschema
import yaml


def is_answered(message):
    """Check whether results exist."""
    return bool(message['results'])


def kgraph_is_weighted(message):
    """Check whether knowledge graph edges have weights.

    Only valid if message has local knowledge graph.
    """
    return all('weight' in edge for edge in message['knowledge_graph']['edges'])


def answers_are_scored(message):
    """Check whether answers have scores.

    Only valid if answers exist.
    """
    if not is_answered(message):
        raise ValueError('Message has no answers.')
    return all('score' in answer for answer in message['results'])
