from ranker.shared.cache import Cache

def test_get_pcounts():
    keys=['CHEBI:92293', 'baddie',  'MONDO:0005249']
    cache = Cache()
    query = "as x MATCH (q:CURIE {concept:x}) return q.concept, q.publication_count"
    result = cache.mquery(keys, 'OMNICORP', query)
    assert "baddie" not in result
    assert result["CHEBI:92293"] == 9
    assert result["MONDO:0005249"] == 128504