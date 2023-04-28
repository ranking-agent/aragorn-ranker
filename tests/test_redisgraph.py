from ranker.shared.cache import Cache

def test_get_pcounts():
    keys=['CHEBI:92293', 'baddie',  'MONDO:0005249']
    cache = Cache()
    query = "as x MATCH (q:CURIE {concept:x}) return q.concept, q.publication_count"
    result = cache.mquery(keys, 'OMNICORP', query)
    assert "baddie" not in result
    assert result["CHEBI:92293"] == 9
    assert result["MONDO:0005249"] == 128504

def test_get_shared_counts():
    keys = [["CHEBI:92293", "MONDO:0005249"], ["baddie", "MONDO:0005249"], ["NCBITaxon:2697049","MONDO:0004730"]]
    cache = Cache()
    results = cache.mquery(keys,"OMNICORP",
              "as q MATCH (a:CURIE {concept:q[0]})-[x]-(b:CURIE {concept:q[1]}) return q[0],q[1],x.publication_count")
    assert ("CHEBI:92293", "MONDO:0005249") not in results
    assert ("baddie", "MONDO:0005249") not in results
    assert results[("NCBITaxon:2697049", "MONDO:0004730")] == 114
