"""Test scoring."""

import pytest
from fastapi.testclient import TestClient
from ranker.server import APP
from ranker.modules.omnicorp_overlay import generate_curie_pairs
from reasoner_pydantic import Response

# this will load all the json test files into global objects to use in a test
from .fixtures import omnicorp_input, property_coalesce, multinodebind_rank

# start a client
client = TestClient(APP)


# def test_omnicorp_overlay(omnicorp_input):
def test_omnicorp_overlay(omnicorp_input):
    """Test that omnicorp_overlay() runs without errors."""

    # Is the input valid
    pydantic_input = Response.parse_obj(omnicorp_input)

    response = client.post("/omnicorp_overlay", json=omnicorp_input)
    assert response.status_code == 200

    # load the json
    answer = response.json()

    # make sure the there are the same number of results that went in
    assert len(answer["message"]["results"]) == len(
        omnicorp_input["message"]["results"]
    )

    # There should be new edges
    assert len(answer["message"]["knowledge_graph"]["edges"]) > len(
        omnicorp_input["message"]["knowledge_graph"]["edges"]
    )

    # assert there are node bindings
    assert len(answer["message"]["results"][0]["node_bindings"]) == 2

    # assert that there are now two auxiliary graphs
    assert (len(answer["message"]["auxiliary_graphs"])) == 2

    # assert that the analysis support is in auxiliary graphs
    omnicorp_graph = answer["message"]["results"][0]["analyses"][0]["support_graphs"][0]
    assert omnicorp_graph in answer["message"]["auxiliary_graphs"]

    # assert that the edges in the omnicorp support graph are the right # and the right predicate
    # there are counts for every pair in this set of 3 curies in the test data.
    omnicorp_edges = answer["message"]["auxiliary_graphs"][omnicorp_graph]["edges"]
    assert len(omnicorp_edges) == 3
    for omni_edge in omnicorp_edges:
        assert (
            answer["message"]["knowledge_graph"]["edges"][omni_edge]["predicate"]
            == "biolink:occurs_together_in_literature_with"
        )


@pytest.mark.asyncio
async def test_generate_node_pairs(omnicorp_input):
    answers = omnicorp_input["message"]["results"]
    x = await generate_curie_pairs(
        answers,
        [],
        {"CHEBI:8871": 10, "MONDO:0004995": 10, "NCBIGene:7124": 10},
        omnicorp_input["message"],
    )
    assert len(x) > 0


@pytest.mark.asyncio
async def test_generate_node_pairs_2(multinodebind_rank):
    answers = multinodebind_rank["message"]["results"]
    counts = {x: 10 for x in multinodebind_rank["message"]["knowledge_graph"]["nodes"]}
    x = await generate_curie_pairs(answers, [], counts, multinodebind_rank["message"])
    assert len(x) > 0


def test_multi_omnicorp_overlay(multinodebind_rank):
    """Test that omnicorp_overlay() runs without errors."""

    # Is the input valid
    pydantic_input = Response.parse_obj(multinodebind_rank)

    response = client.post("/omnicorp_overlay", json=multinodebind_rank)
    assert response.status_code == 200

    # load the json
    answer = response.json()

    # make sure the there are the same number of results that went in
    assert len(answer["message"]["results"]) == len(
        multinodebind_rank["message"]["results"]
    )

    # There should be 2 new edges
    assert len(answer["message"]["knowledge_graph"]["edges"]) == 3 + len(
        multinodebind_rank["message"]["knowledge_graph"]["edges"]
    )

    # assert there are node bindings
    assert len(answer["message"]["results"][0]["node_bindings"]) == 2

    # assert that there is now 1 auxiliary graphs
    assert (len(answer["message"]["auxiliary_graphs"])) == 1

    # assert that the analysis support is in auxiliary graphs
    omnicorp_graph = answer["message"]["results"][0]["analyses"][0]["support_graphs"][0]
    assert omnicorp_graph in answer["message"]["auxiliary_graphs"]

    # assert that the edges in the omnicorp support graph are the right # and the right predicate
    # there are counts for every pair in this set of 3 curies in the test data.
    omnicorp_edges = answer["message"]["auxiliary_graphs"][omnicorp_graph]["edges"]
    assert len(omnicorp_edges) == 3
    for omni_edge in omnicorp_edges:
        assert (
            answer["message"]["knowledge_graph"]["edges"][omni_edge]["predicate"]
            == "biolink:occurs_together_in_literature_with"
        )
