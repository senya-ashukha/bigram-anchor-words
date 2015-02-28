import logging
import numpy as np

from tmtk.utils.math import norn_mtx

def plsa_model(documents_train, documents_test, num_topics=100, num_itter=10, metric=None, verbose=False):
    doc_count, wrd_count = len(documents_train), len(set([wrd for doc in documents_train for wrd, _ in doc]))
    F, T = norn_mtx(wrd_count, num_topics, axis='x'), norn_mtx(num_topics, doc_count, axis='y')

    for itter in xrange(num_itter):
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

        if verbose:
            metric_val = metric(F, T, documents_train, documents_test)
            logging.info('itter %s: %s' % (itter, metric_val))

    if metric and verbose:
        metric_val = metric(F, T, documents_train, documents_test)
        logging.info('%s: %s' % (metric.__name__, metric_val))

    return F, T