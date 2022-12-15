
import math

SOURCE_WEIGHTS = {
    "infores:automat-pharos": 0.7,
    "infores:aragorn-ranker-ara": 0.25,
    "infores:semmeddb": 0.05,
}

SOURCE_STEEPNESS = {
    "infores:automat-pharos": 0.549306144334,
    "infores:aragorn-ranker-ara": 0.549306144334,
    "infores:semmeddb": 0.549306144334,
}

def source_weight(source):
    return SOURCE_WEIGHTS.get(source, 0) 

def source_sigmoid(source, effective_pubs):
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
    steepness = SOURCE_STEEPNESS.get(source, 0)
    return 2 / (1 + math.exp(-steepness * max(effective_pubs, 0))) - 1
