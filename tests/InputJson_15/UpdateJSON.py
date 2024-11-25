import os
import json

# TO run this, grab the jsons you want from earlier version, copy them into this directory (changing the names to
# whatever_original.json) and then run it.  There some cleaner ways to do this...


def get_originals():
    files = os.listdir(".")
    originals = []
    for f in files:
        if f.endswith("_original.json"):
            originals.append(f)
    return originals


def go():
    originals = get_originals()
    for original in originals:
        parse(original)


def add_logs(trapi):
    if "logs" not in trapi.keys():
        trapi["logs"] = []


def add_sources_and_attributes_to_edges(trapi):
    for edge_id, edge in trapi["message"]["knowledge_graph"]["edges"].items():
        if "sources" not in edge:
            edge["sources"] = [
                {
                    "resource_id": "infores:madeup",
                    "resource_role": "primary_knowledge_source",
                }
            ]
        if "attributes" not in edge:
            edge["attributes"] = []


def add_attributes_to_nodes(trapi):
    for node_id, node in trapi["message"]["knowledge_graph"]["nodes"].items():
        if "categories" not in node:
            node["categories"] = ["biolink:NamedThing"]
        if "attributes" not in node:
            node["attributes"] = []


def add_attributes_to_bindings(trapi):
    for result in trapi["message"]["results"]:
        for node_binding, nb in result["node_bindings"].items():
            for nbx in nb:
                if "attributes" not in nbx:
                    nbx["attributes"] = []
        for analysis in result["analyses"]:
            for edge_binding, eb in analysis["edge_bindings"].items():
                for ebx in eb:
                    if "attributes" not in ebx:
                        ebx["attributes"] = []


def add_attributes_to_auxgraphs(trapi):
    if "auxiliary_graphs" not in trapi["message"]:
        return
    for auxid, auxgraph in trapi["message"]["auxiliary_graphs"].items():
        if "attributes" not in auxgraph:
            auxgraph["attributes"] = []


def fix_analysis(trapi):
    for result in trapi["message"]["results"]:
        if "analysis" not in result:
            result["analyses"] = []
        if "edge_bindings" in result:
            analysis = {
                "edge_bindings": result["edge_bindings"],
                "attributes": [],
                "resource_id": "fake:thing",
            }
            result["analyses"].append(analysis)
            del result["edge_bindings"]


def parse(original):
    with open(original, "r") as inf:
        trapi = json.load(inf)
    add_logs(trapi)
    add_sources_and_attributes_to_edges(trapi)
    add_attributes_to_nodes(trapi)
    fix_analysis(trapi)
    add_attributes_to_bindings(trapi)
    add_attributes_to_auxgraphs(trapi)
    newname = original[: -len("_original.json")] + ".json"
    with open(newname, "w") as outf:
        json.dump(trapi, outf, indent=2)


if __name__ == "__main__":
    go()
