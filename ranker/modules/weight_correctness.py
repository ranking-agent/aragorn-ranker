"""Weight edges."""
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

from fastapi import Query
from fastapi.responses import JSONResponse
from reasoner_pydantic import Response as PDResponse

from ranker.shared.sources import source_sigmoid, DEFAULT_SOURCE_STEEPNESS
from ranker.shared.util import create_log_entry

logger = logging.getLogger(__name__)

async def query(
    request: PDResponse,
    relevance: Optional[float] = Query(
        0.0025,
        description="Portion of cooccurrence publications relevant to a question",
    )
):
    """Weight kgraph edges based on metadata.

    "19 pubs from CTD is a 1, and 2 should at least be 0.5"
        - cbizon
    """

    # get the debug environment variable
    debug = os.environ.get("DEBUG_TIMING", "False")

    if debug == "True":
        dt_start = datetime.now()

    # save the message
    in_message = request.dict()

    # save the logs for the response (if any)
    if "logs" not in in_message or in_message["logs"] is None:
        in_message["logs"] = []
    else:
        # these timestamps are causing json serialization issues
        # so here we convert them to strings.
        for log in in_message["logs"]:
            log["timestamp"] = str(log["timestamp"])

    # init the status code
    status_code: int = 200

    message = in_message["message"]

    try:
        correct_weights(message, relevance)
    except Exception as e:
        # put the error in the response
        status_code = 500
        logger.exception(f"Aragorn-ranker/weight correctness exception {e}")
        # save any log entries
        # in_message['logs'].append(create_log_entry(f'Exception: {str(e)}', 'ERROR'))

    if debug == "True":
        diff = datetime.now() - dt_start
        in_message["logs"].append(
            create_log_entry(
                f"End of weight correctness processing. Time elapsed: {diff.seconds} seconds",
                "DEBUG",
            )
        )

    # return the result to the caller
    return JSONResponse(content=in_message, status_code=status_code)

def correct_weights(message, relevance=0.0025, source_steepness=DEFAULT_SOURCE_STEEPNESS):
    # constant count of all publications
    all_pubs = 27840000

    # get the data nodes we need
    results = message["results"]
    kgraph = message["knowledge_graph"]

    # storage for the publication counts for the node
    node_pubs: dict = {}

    # for each node in the knowledge graph
    for n in kgraph["nodes"]:
        # init the count value
        omnicorp_article_count: int = 0

        # get the article count atribute
        for p in kgraph["nodes"][n]["attributes"]:
            # is this what we are looking for
            if p["original_attribute_name"] == "omnicorp_article_count":
                # save it
                omnicorp_article_count = p["value"]

                # no need to continue
                break

        # add the node d and count to the dict
        node_pubs.update({n: omnicorp_article_count})

    # map kedges to result edge bindings
    krmap = defaultdict(list)

    # for each result listed in the data get a map reference and default the weight attribute
    for result in results:
        # for every edge binding result
        for eb in result["edge_bindings"]:
            # loop through the edge binding
            for idx, binding_val in enumerate(result["edge_bindings"][eb]):
                # get a reference to the weight for easy update later
                ebi = result["edge_bindings"][eb][idx]
                found = False

                # is there already a list of attributes
                if "attributes" in ebi and ebi["attributes"] is not None:
                    # loop through the attributes
                    for item in ebi["attributes"]:
                        # search for the weight attribute
                        if item["original_attribute_name"].startswith("weight"):
                            found = True

                            break

                # was the attribute found
                if not found:
                    if "attributes" not in ebi or ebi["attributes"] is None:
                        ebi["attributes"] = []

                    # create an Attribute
                    ebi["attributes"].append(
                        {
                            "original_attribute_name": "weight",
                            "attribute_type_id": "biolink:has_numeric_value",
                            "value": 1,
                            "value_type_id": "EDAM:data_1669",
                        }
                    )
                krmap[binding_val["id"]].append(ebi)

    # get the knowledge graph edges
    edges = kgraph["edges"]

    # for each knowledge graph edge
    for edge in edges:
        # We are getting some results back (BTE?) that have "publications": ['PMID:1234|2345|83984']
        attributes = edges[edge].get("attributes", None)

        # init storage for the publications and their count
        publications = []
        num_publications = 0

        if attributes is not None:
            # for each data attribute collect the needed params
            for attribute in attributes:
                # This picks up omnicorp
                if attribute.get("original_attribute_name", None) is not None:
                    # is this the publication list
                    if attribute["original_attribute_name"].startswith(
                        "publications"
                    ):
                        publications = attribute["value"]
                    # else is this the number of publications
                    elif attribute["original_attribute_name"].startswith(
                        "num_publications"
                    ):
                        num_publications = attribute.get("value", 0)
                # This picks up Text Miner KP
                elif (
                    attribute["attribute_type_id"] == "biolink:supporting_document"
                ):
                    publications = attribute["value"]
                    if isinstance(publications, str):
                        publications = [publications]
                # This picks up how BTE returns pubs
                elif attribute["attribute_type_id"] == "biolink:publications":
                    publications = attribute["value"]

            # Record the source of origination
            edge_info = {
                "biolink:aggregator_knowledge_source": "not_found",
                "biolink:original_knowledge_source": "not_found",
                "biolink:primary_knowledge_source": "not_found",
            }
            for attribute in reversed(attributes):
                if attribute.get("attribute_type_id", None) is not None:
                    if attribute["attribute_type_id"] in edge_info.keys():
                        v = attribute.get("value", None)
                        if type(v) is list:
                            v = v[0]
                        if v is not None:
                            edge_info[attribute["attribute_type_id"]] = v
                        else:
                            edge_info[
                                attribute["attribute_type_id"]
                            ] = "unspecified"

            if edge_info["biolink:original_knowledge_source"] != "not_found":
                edge_info_final = edge_info["biolink:original_knowledge_source"]
            elif edge_info["biolink:primary_knowledge_source"] != "not_found":
                edge_info_final = edge_info["biolink:primary_knowledge_source"]
            elif edge_info["biolink:aggregator_knowledge_source"] != "not_found":
                edge_info_final = edge_info["biolink:aggregator_knowledge_source"]
            else:
                edge_info_final = None

            # if there was only 1 publication value found insure it wasnt a character separated list
            if len(publications) == 1:
                if "|" in publications[0]:
                    publications = publications[0].split("|")
                elif "," in publications[0]:
                    publications = publications[0].split(",")

                # get the real publication count
                num_publications = len(publications)

            # if there was no publication count found revert to the number of individual values
            if num_publications == 0:
                num_publications = len(publications)

            if (
                edges[edge].get("predicate")
                == "biolink:occurs_together_in_literature_with"
            ):
                subject_pubs = int(node_pubs[edges[edge]["subject"]])
                object_pubs = int(node_pubs[edges[edge]["object"]])

                cov = (num_publications / all_pubs) - (subject_pubs / all_pubs) * (
                    object_pubs / all_pubs
                )
                cov = max((cov, 0.0))
                effective_pubs = cov * all_pubs * relevance
            else:
                effective_pubs = num_publications + 1  # consider the curation a pub

            # if there is something to add this new attribute to
            for edgebinding in krmap[edge]:
                # is there already a list of attributes
                if "attributes" in edgebinding:
                    # loop through the attributes
                    for item in edgebinding["attributes"]:
                        # search for the weight attribute
                        if item["original_attribute_name"].startswith("weight"):
                            # update the params
                            item["attribute_type_id"] = "biolink:has_numeric_value"
                            item["value"] = item["value"] * source_sigmoid(edge_info_final, effective_pubs, source_steepness=source_steepness)
                            item["value_type_id"] = "EDAM:data_1669"
                            if edge_info_final is not None:
                                if (
                                    "attributes" not in item
                                    or item["attributes"] is None
                                ):
                                    item["attributes"] = []

                                item["attributes"].append(
                                    {
                                        "original_attribute_name": "aragorn_weight_source",
                                        "attribute_type_id": "biolink:has_qualitative_value",
                                        "value": edge_info_final,
                                        "value_type_id": "biolink:InformationResource",
                                    }
                                )
                            found = True
                            break

    # save the new knowledge graph data
    message["knowledge_graph"] = kgraph