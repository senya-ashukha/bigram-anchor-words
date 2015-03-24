import numpy as np

from operator import itemgetter
from tmtk.utils.math import norn_mtx

def plsa_model(documents_train, documents_test, wrd_count, num_topics=100, num_iter=10, metric=None, verbose=False):
    doc_count = len(documents_train)
    F, T = norn_mtx(wrd_count, num_topics, axis='x'), norn_mtx(num_topics, doc_count, axis='y')

    for itter in xrange(num_iter):
        Nwt, Ntd = np.zeros((wrd_count, num_topics)), np.zeros((num_topics, doc_count))
        Nt, Nd = np.zeros(num_topics), np.zeros(doc_count)

        for d in xrange(doc_count):
            for w, ndw in documents_train[d]:
                ndwt = F[w, :] * T[:, d]
                ndwt *= ndw * (1.0 / ndwt.sum())

                Nwt[w] += ndwt
                Ntd[:, d] += ndwt
                Nt += ndwt
                Nd[d] += ndwt.sum()

        for w in xrange(wrd_count):
            F[w] = Nwt[w] / Nt

        for t in range(num_topics):
            T[t] = Ntd[t] / Nd

        if metric and verbose:
            metric_val = metric(F, documents_train, documents_test)
            print 'iter %s: %s' % (itter, metric_val)

    if metric:
        metric_val = metric(F, documents_train, documents_test)
        print 'end: %s' % metric_val

    return F, T

def print_topics(F, id_to_wrd, top=8):
    for i, column in enumerate(F.T):
        col = list(enumerate(column))
        col = map(itemgetter(0), sorted(col, key=itemgetter(1), reverse=True)[:top])
        col = 'Topic %s: ' % i + ' '.join(map(lambda x: id_to_wrd[x], col))

        print col


