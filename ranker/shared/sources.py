from cmath import exp
import math

# Commenting out b/c symbol seems to be missing and parameters is unused?
# from symbol import parameters

DEFAULT_WEIGHT = 1e-2

BLENDED_PROFILE = {
    "source_weights": {
        "infores:omnicorp": {
            "literature_co-occurrence": 1,
        },
        "infores:text-mining-provider-targeted": {
            "publications": 0.5,
            "confidence_score": 1,
        },
        "infores:genetics-data-provider": {"p_value": 1},
        "infores:icees-kg": {"p_value": 1},
        "infores:gwas-catalog": {"p_value": 1},
    },
    "source_transformation": {
        "infores:omnicorp": {
            "literature_co-occurrence": {
                "lower": -1,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.00033,
            }
        },
        "infores:text-mining-provider-targeted": {
            "publications": {"lower": -1, "upper": 1, "midpoint": 0, "rate": 4e-4},
            "confidence_score": {"lower": -1, "upper": 1, "midpoint": 0, "rate": 3},
        },
        "infores:genetics-data-provider": {
            "p_value": {"lower": 0, "upper": 1, "midpoint": 0.05, "rate": -150}
        },
        "infores:icees-kg": {
            "p_value": {"lower": 0, "upper": 1, "midpoint": 0.05, "rate": -150}
        },
        "infores:gwas-catalog": {
            "p_value": {"lower": 0, "upper": 1, "midpoint": 0.05, "rate": -150}
        },
    },
    "unknown_source_weight": {
        "publications": 1,
        "literature_co-occurrence": 1,
        "affinity": 1,
        "unknown_property": 0,
    },
    "unknown_source_transformation": {
        "publications": {"lower": -1, "upper": 1, "midpoint": 0, "rate": 0.574213221},
        "literature_co-occurrence": {
            "lower": -1,
            "upper": 1,
            "midpoint": 0,
            "rate": 0.001373265360835,
        },
        "affinity": {"lower": -1, "upper": 1, "midpoint": 0, "rate": 0.4},
        "unknown_property": {"lower": 0, "upper": 0, "midpoint": 0, "rate": 0},
    },
    "base_weights": {
        "default_weight": 1e-2,
        "infores:omnicorp": 0,
        "infores:drugcentral": 5e-2,
        "infores:hetionet": 3e-2,
        "infores:text-mining-provider-targeted": 5e-3,
        "infores:icees-kg": 3e-2,
        "infores:gwas-catalog": 3e-2,
    },
    "omnicorp_relevence": 0.0025,
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
        },
    },
    "source_transformation": {
        "infores:automat-pharos": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
        "infores:aragorn-ranker-ara": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
        "infores:semmeddb": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
    },
    "unknown_source_weight": {"publications": 1, "unknown_property": 0},
    "unknown_source_transformation": {
        "publications": {"lower": 0, "upper": 1, "midpoint": 0, "rate": 0.574213221},
        "p_value": {"lower": 1, "upper": 0, "midpoint": 0.055, "rate": 200.574213221},
        "unknown_property": {
            "lower": 0,
            "upper": 1,
            "midpoint": 0,
            "rate": 0.574213221,
        },
    },
    "base_weights": {"default_weight": 1e-2},
    "omnicorp_relevence": 0.0025,
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
        },
    },
    "source_transformation": {
        "infores:automat-pharos": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
        "infores:aragorn-ranker-ara": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
        "infores:semmeddb": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
    },
    "unknown_source_weight": {"publications": 1, "unknown_property": 0},
    "unknown_source_transformation": {
        "publications": {"lower": 0, "upper": 1, "midpoint": 0, "rate": 0.574213221},
        "p_value": {"lower": 1, "upper": 0, "midpoint": 0.055, "rate": 200.574213221},
        "unknown_property": {
            "lower": 0,
            "upper": 1,
            "midpoint": 0,
            "rate": 0.574213221,
        },
    },
    "base_weights": {"default_weight": 1e-2},
    "omnicorp_relevence": 0.0025,
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
        },
    },
    "source_transformation": {
        "infores:automat-pharos": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
        "infores:aragorn-ranker-ara": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
        "infores:semmeddb": {
            "publications": {
                "lower": 0,
                "upper": 1,
                "midpoint": 0,
                "rate": 0.574213221,
            },
            "p_value": {
                "lower": 1,
                "upper": 0,
                "midpoint": 0.055,
                "rate": 200.574213221,
            },
        },
    },
    "unknown_source_weight": {"publications": 1, "unknown_property": 0},
    "unknown_source_transformation": {
        "publications": {"lower": 0, "upper": 1, "midpoint": 0, "rate": 0.574213221},
        "p_value": {"lower": 1, "upper": 0, "midpoint": 0.055, "rate": 200.574213221},
        "unknown_property": {
            "lower": 0,
            "upper": 1,
            "midpoint": 0,
            "rate": 0.574213221,
        },
    },
    "base_weights": {"default_weight": 1e-2},
    "omnicorp_relevence": 0.0025,
}


def get_base_weight(source, base_weights=BLENDED_PROFILE["base_weights"]):
    return base_weights.get(source, base_weights.get("default_weight", DEFAULT_WEIGHT))


def get_source_weight(
    source,
    property,
    source_weights=BLENDED_PROFILE["source_weights"],
    unknown_source_weight=BLENDED_PROFILE["unknown_source_weight"],
):
    return source_weights.get(source, unknown_source_weight).get(
        property, unknown_source_weight["unknown_property"]
    )


def get_source_sigmoid(
    value,
    source="unknown",
    property="unknown",
    source_transformation=BLENDED_PROFILE["source_transformation"],
    unknown_source_transformation=BLENDED_PROFILE["unknown_source_transformation"],
):
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
    parameters = source_transformation.get(source, unknown_source_transformation).get(
        property, unknown_source_transformation["unknown_property"]
    )
    upper = parameters["upper"]
    lower = parameters["lower"]
    midpoint = parameters["midpoint"]
    rate = parameters["rate"]
    return lower + ((upper - lower) / (1 + math.exp(-1 * rate * (value - midpoint))))


def get_profile(profile="blended"):
    if profile == "clinical":
        source_weights = CLINICAL_PROFILE["source_weights"]
        unknown_source_weight = CLINICAL_PROFILE["unknown_source_weight"]
        source_transformation = CLINICAL_PROFILE["source_transformation"]
        unknown_source_transformation = CLINICAL_PROFILE[
            "unknown_source_transformation"
        ]
        base_weights = CLINICAL_PROFILE["base_weights"]
    elif profile == "correlated":
        source_weights = CORRELATED_PROFILE["source_weights"]
        unknown_source_weight = CORRELATED_PROFILE["unknown_source_weight"]
        source_transformation = CORRELATED_PROFILE["source_transformation"]
        unknown_source_transformation = CORRELATED_PROFILE[
            "unknown_source_transformation"
        ]
        base_weights = CORRELATED_PROFILE["base_weights"]

    elif profile == "curated":
        source_weights = CURATED_PROFILE["source_weights"]
        unknown_source_weight = CURATED_PROFILE["unknown_source_weight"]
        source_transformation = CURATED_PROFILE["source_transformation"]
        unknown_source_transformation = CURATED_PROFILE["unknown_source_transformation"]
        base_weights = CURATED_PROFILE["base_weights"]

    else:
        source_weights = BLENDED_PROFILE["source_weights"]
        unknown_source_weight = BLENDED_PROFILE["unknown_source_weight"]
        source_transformation = BLENDED_PROFILE["source_transformation"]
        unknown_source_transformation = BLENDED_PROFILE["unknown_source_transformation"]
        base_weights = BLENDED_PROFILE["base_weights"]

    return (
        source_weights,
        unknown_source_weight,
        source_transformation,
        unknown_source_transformation,
        base_weights,
    )
