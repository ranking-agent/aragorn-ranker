"""Reasoner filters utilities."""
import random
import string


def random_string(length=10):
    """Return a random N-character-long string."""
    return ''.join(random.choice(string.ascii_lowercase) for x in range(length))


def argsort(x, reverse=False):
    """Return the indices that would sort the array."""
    return [p[0] for p in sorted(enumerate(x), key=lambda elem: elem[1], reverse=reverse)]


def flatten_semilist(x):
    """Convert a semi-nested list - a list of (lists and scalars) - to a flat list."""
    # convert to a list of lists
    lists = [n if isinstance(n, list) else [n] for n in x]
    # flatten nested list
    return [e for el in lists for e in el]


def batches(arr, n):
    """Iterate over arr by batches of size n."""
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def get_curie_prefix(curie):
    if ':' not in curie:
        raise ValueError('Curies ought to contain a colon')
    return curie.upper().split(':')[0]
