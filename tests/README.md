# Testing Aragorn-ranker

### Test Files

The aragorn-ranker exposes three endpoints, and there is a pytest module for each:

* [`test_omnicorp_overlay.py`](test_omnicorp_overlay.py):

  We check the overall functioning of the overlay service, making sure that the expected number of edges are added to a particular input graph.

* [`test_correctness.py`](test_correctness.py):

  Test that weight() runs without errors and that the weights are correctly ordered.  i.e. that better supported edges have higher weights.

* [`test_score.py`](test_score.py):

  Check that the scoring function returns correctly for typical inputs, and that the score is nonzero for specific configurations.

### Workflow

Tests are run automatically via GitHub Actions whenever a pull request review is requested.


