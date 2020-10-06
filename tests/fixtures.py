"""Test message formatting."""
import glob
import json
import os

import pytest

# generate fixture for each JSON file in tests/data
files = glob.glob('tests/data/*.json')
fixtures = []
for filename in files:
    base = os.path.splitext(os.path.basename(filename))[0]

    @pytest.fixture(name=base, scope='module')
    def fixture(filename=filename):
        """Get message with ambiguous kgraph."""
        with open(filename, 'r') as stream:
            return json.load(stream)
    globals()[base] = fixture
