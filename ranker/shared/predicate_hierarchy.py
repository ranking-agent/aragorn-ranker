from typing import Tuple
from bmt.toolkit import Toolkit as BMToolkit

bmt = BMToolkit()


def find_height(children):
    height = -1
    for child in children:
        grandchildren = bmt.get_children(child)
        height = max(height, find_height(grandchildren))
    return height + 1


def find_predicate_heirarchy(predicate) -> Tuple[int]:
    """find height, depth and number of decendants of a biolink predicate

    Args:
        predicate (str): a biolink predicate
    Returns:
        tuple[int]: height, depth, num_descendents
    """

    descendents = bmt.get_descendants(predicate)[1:]
    ancestors = bmt.get_ancestors(predicate)[1:]

    height = find_height(bmt.get_children(predicate))
    depth = len(ancestors)
    num_descendents = len(descendents)
    print(height, depth, num_descendents)


if __name__ == "__main__":
    find_predicate_heirarchy("associated_with")