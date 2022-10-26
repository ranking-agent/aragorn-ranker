
from cmath import exp
import math
from symbol import parameters

BLENDED_PROFILE = {
    "source_weights": {
        "infores:automat-pharos": {
            "publications": 0.7,
        },
        "infores:aragorn-ranker-ara": {
            "publications": 0.25,
        },
        "infores:semmeddb": {
            "publications": 0.05,
        }
    },
    "source_transformation": {
        "infores:automat-pharos": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:aragorn-ranker-ara": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:semmeddb": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        }
    },
    "unknown_source_weight": {
        "publicaitons": 1,
        "unknown_property" : 0
    },
    "unknown_source_transformation": {
        "publications": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        },
        "unkown_property": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        }
    }
}

CURATED_PROFILE = {
    "source_weights": {
        "infores:automat-pharos": {
            "publications": 0.7,
        },
        "infores:aragorn-ranker-ara": {
            "publications": 0.25,
        },
        "infores:semmeddb": {
            "publications": 0.05,
        }
    },
    "source_transformation": {
        "infores:automat-pharos": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:aragorn-ranker-ara": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:semmeddb": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        }
    },
    "unknown_source_weight": {
        "publicaitons": 1,
        "unknown_property" : 0
    },
    "unknown_source_transformation": {
        "publications": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        },
        "unkown_property": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        }
    }
}

CORRELATED_PROFILE = {
    "source_weights": {
        "infores:automat-pharos": {
            "publications": 0.7,
        },
        "infores:aragorn-ranker-ara": {
            "publications": 0.25,
        },
        "infores:semmeddb": {
            "publications": 0.05,
        }
    },
    "source_transformation": {
        "infores:automat-pharos": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:aragorn-ranker-ara": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:semmeddb": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        }
    },
    "unknown_source_weight": {
        "publicaitons": 1,
        "unknown_property" : 0
    },
    "unknown_source_transformation": {
        "publications": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        },
        "unkown_property": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        }
    }
}

CLINICAL_PROFILE = {
    "source_weights": {
        "infores:automat-pharos": {
            "publications": 0.7,
        },
        "infores:aragorn-ranker-ara": {
            "publications": 0.25,
        },
        "infores:semmeddb": {
            "publications": 0.05,
        }
    },
    "source_transformation": {
        "infores:automat-pharos": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:aragorn-ranker-ara": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        },
        "infores:semmeddb": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "start": 0,
                "initial_value": 1,
                "explanatory_value": .1,
                "response_value": .5
            }
        }
    },
    "unknown_source_weight": {
        "publicaitons": 1,
        "unknown_property" : 0
    },
    "unknown_source_transformation": {
        "publications": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        },
        "unkown_property": {
            "lower": 0,
            "upper": 1,
            "start": 0,
            "initial_value": 1,
            "explanatory_value": .1,
            "response_value": .5
        }
    }
}

def source_weight(source, property, source_weights = BLENDED_PROFILE["source_weights"], unknown_source_weight = BLENDED_PROFILE["unknown_source_weight"]):
    return source_weights.get(source, unknown_source_weight).get(property, unknown_source_weight["unknown_property"])

def source_sigmoid(source, property, value, source_transformation = BLENDED_PROFILE["source_transformation"], unknown_source_transformation = BLENDED_PROFILE["source_transformation"]):
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
    parameters = source_transformation.get(source, unknown_source_transformation).get(property, unknown_source_transformation["unknown_property"])
    upper = parameters["upper"]
    lower = parameters["lower"]
    initial = parameters["initial_value"]
    start = parameters["start"]
    explanatory = parameters["explanatory_value"]
    response = parameters["response_value"]
    rate = (math.log((upper - response) / (initial * (response - lower))) / (math.log(math.e) * (explanatory - start))) * -1
    return lower + ((upper - lower) / (1 + initial * exp(-1 * rate * (value - start))))

def get_profile(profile = "blended"):
    if profile == "clinical":
        source_weights = CLINICAL_PROFILE["source_weights"]
        unknown_source_weight = CLINICAL_PROFILE["unknown_source_weight"]
        source_transformation = CLINICAL_PROFILE["source_transformation"]
        unknown_source_transformation = CLINICAL_PROFILE["unknown_source_transformation"]
    elif profile == "correlated":
        source_weights = CORRELATED_PROFILE["source_weights"]
        unknown_source_weight = CORRELATED_PROFILE["unknown_source_weight"]
        source_transformation = CORRELATED_PROFILE["source_transformation"]
        unknown_source_transformation = CORRELATED_PROFILE["unknown_source_transformation"]
    elif profile == "curated":
        source_weights = CURATED_PROFILE["source_weights"]
        unknown_source_weight = CURATED_PROFILE["unknown_source_weight"]
        source_transformation = CURATED_PROFILE["source_transformation"]
        unknown_source_transformation = CURATED_PROFILE["unknown_source_transformation"]
    else:
        source_weights = BLENDED_PROFILE["source_weights"]
        unknown_source_weight = BLENDED_PROFILE["unknown_source_weight"]
        source_transformation = BLENDED_PROFILE["source_transformation"]
        unknown_source_transformation = BLENDED_PROFILE["unknown_source_transformation"]
    
    return source_weights, unknown_source_weight, source_transformation, unknown_source_transformation