import math

import numpy as np


#Ti, docs = estimate_teta(F, documents_test)
#print '%s: train = %.2f, test = %.2f' % (metric.__name__[:4], metric(F, T, documents_train), metric(F, Ti, docs))

def preplexity(F, T, documents):
    doc_count, wrd_count = len(T[0]), len(F[0])
    lh = 0
    for d in xrange(doc_count):
        for w, ndw in documents[d]:
            lh += ndw * math.log(np.dot(F[w, :], T[:, d]))
    return math.e ** (-lh / sum([sum([ndw for _, ndw in documents[d]]) for d in xrange(doc_count)]))
