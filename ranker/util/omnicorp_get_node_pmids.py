"""Literature related to node."""
from fastapi.responses import JSONResponse
#from ranker.shared.omnicorp import OmnicorpSupport
from ranker.shared.util import Curie


async def get_node_pmids(curies: Curie):
    """Find the pubmed ids where the specified node appears.

    inputs: curies (CurieList): a dictionary contianing the key "curies"
            whose value is a list of curie-strings (see util.py). The length of
            this list is exactly one.
            Example:
            curies = {
                "curies": ['MESH:D014867']
            }
    """
    nodeList = curies.curies
    if len(nodeList) != 1:
        raise ValueError(f'nodeList must contain exactly 1 node: nodeList = {nodeList}')

    #The new omnicorp doesn't support this yet, so we'll just return an empty list
    #it would be straightforward to implement, but it would require adding the pmids to the redisgraph

    #async with OmnicorpSupport() as supporter:
    #    shared_pubmed_ids = await supporter.node_pmids(nodeList[0])
    #    status_code = 200
    shared_pubmed_ids = []
    status_code = 200

    # return the result to the caller
    return JSONResponse(content=shared_pubmed_ids, status_code=status_code)
