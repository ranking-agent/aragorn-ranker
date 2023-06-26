from ranker.shared.cache import Cache

def test_get_counts():
    keys=['CHEBI:92293', 'baddie',  'MONDO:0005249', 'NCBITaxon:2697049', 'MONDO:0004730']
    cache = Cache()
    result = cache.curie_query(keys)
    assert result["baddie"] == {}
    assert result["CHEBI:92293"]["pmc"] == 9
    assert result["MONDO:0005249"]["pmc"] == 128504
    CHEBI_92293_index = int(result["CHEBI:92293"]["index"])
    MONDO_0005249_index = int(result["MONDO:0005249"]["index"])
    NCBI_2697049_index = int(result["NCBITaxon:2697049"]["index"])
    MONDO_0004730_index = int(result["MONDO:0004730"]["index"])
    #with the index thing, there's no way to have a key that doesn't have a valid index in it
    #keys = [["CHEBI:92293", "MONDO:0005249"], ["baddie", "MONDO:0005249"], ["NCBITaxon:2697049","MONDO:0004730"]]
    pair1 = (CHEBI_92293_index, MONDO_0005249_index)
    pair2 = (NCBI_2697049_index, MONDO_0004730_index)
    key1 = f"{min(pair1)}_{max(pair1)}"
    key2 = f"{min(pair2)}_{max(pair2)}"
    keys = [key1, key2]
    results = cache.shared_count_query(keys)
    assert results[key1] == 0
    assert results[key2] == 114
