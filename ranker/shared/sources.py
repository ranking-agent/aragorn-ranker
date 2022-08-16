
import math

DEFAULT_SOURCE_WEIGHTS = {
    # "infores:automat-pharos": 0.7,
    # "infores:aragorn-ranker-ara": 0.25,
    # "infores:semmeddb": 0.05,
}

DEFAULT_SOURCE_STEEPNESS = {
    # "infores:automat-pharos": 0.549306144334,
    # "infores:aragorn-ranker-ara": 0.549306144334,
    # "infores:semmeddb": 0.549306144334,
}

UNKNOWN_SOURCE_WEIGHT = 1
UNKNOWN_SOURCE_STEEPNESS = 0.549306144334

def source_weight(source, source_weights = DEFAULT_SOURCE_WEIGHTS, unknown_source_weight = UNKNOWN_SOURCE_WEIGHT):
    return source_weights.get(source, unknown_source_weight) 

def source_sigmoid(source, effective_pubs, source_steepness = DEFAULT_SOURCE_STEEPNESS, unknown_source_steepness = UNKNOWN_SOURCE_STEEPNESS):
    """
    0-centered sigmoid used to map the number of publications found by a source
    to its weight. For all unknown sources, this function evaluates to 0.

    Args:
        num_pubs (int): Number of publications from source.
        steepness (float): Parameter in [0, inf) that specifies the steepness
            of the sigmoid.
    
    Returns:
        Weight associated with `effective_pubs` publications (using sigmoid)
    """
    steepness = source_steepness.get(source, unknown_source_steepness)
    return 2 / (1 + math.exp(-steepness * max(effective_pubs, 0))) - 1
