from itertools import izip_longest


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def all_pairs(iter_obj):
    for i, elem1 in enumerate(iter_obj):
        for j, elem2 in enumerate(iter_obj):
            if i < j:
                yield (elem1, elem2)