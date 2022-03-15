"""Literature co-occurrence support."""
from fastapi.responses import JSONResponse
from ranker.shared.omnicorp import OmnicorpSupport
from ranker.shared.util import CurieList


async def shared_pmids(curies: CurieList):
    """Find the pubmed ids where node1 and node2 co-occur.

    inputs: curies (CurieList): a dictionarie contianing the key "curies"
            whose value is a list of curie-strings (see util.py).
            Example:
            curies = {
                "curies": ['MESH:D014867', 'NCIT:C34373']
            }
    """
    nodeList = curies.curies
    if len(nodeList) != 2:
        raise ValueError(f'nodeList must contain exactly 2 nodes: nodeList = {nodeList}')

    async with OmnicorpSupport() as supporter:
        shared_pubmedids = await supporter.term_to_term_pmids(nodeList[0], nodeList[1])
        status_code = 200

    # return the result to the caller
    return JSONResponse(content=shared_pubmedids, status_code=status_code)
