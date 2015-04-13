import numpy as np

from utils import get_topic
from progress.bar import Bar

from tmtk.utils.math import norn_mtx
from tmtk.utils.logger import get_logger
from tmtk.metrics.utils import estimate_teta_full
from tmtk.collection.collection import bag_of_words

logger = get_logger()

def plsa_model(collection, wrd_count, num_topics=100, num_iter=10, metrics=None, verbose=False, F=None):
    logger.info('Start plsa_model')

    logger.info('Create bag of words')
    bw_train, bw_test = bag_of_words(collection.documents_train), bag_of_words(collection.documents_test)

    doc_count = len(bw_train)
    if F is None:
        F, T = norn_mtx(wrd_count, num_topics, axis='x'), norn_mtx(num_topics, doc_count, axis='y')
    else:
        T = estimate_teta_full(F, bw_train)

    if not(metrics and verbose):
        bar = Bar('Processing', max=num_iter)

    logger.info('Begin itters')
    for itter in xrange(num_iter):
        Nwt, Ntd = np.zeros((wrd_count, num_topics)), np.zeros((num_topics, doc_count))
        Nt, Nd = np.zeros(num_topics), np.zeros(doc_count)

        for d in xrange(doc_count):
            for w, ndw in bw_train[d]:
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

        if metrics and verbose:
            if itter % 2 == 1:
                metric_val = [metric(F, collection.documents_train, collection.documents_test) for metric in metrics]
                print 'iter %s: %s' % (str(itter).zfill(2), ' '.join(metric_val))
        else:
            bar.next()

    if not(metrics and verbose):
        bar.finish()

    if metrics:
        logger.info('Eval metrics')
        metric_val = [metric(F, collection.documents_train, collection.documents_test) for metric in metrics]
        print 'end: %s' % ' '.join(metric_val)

    return F, T

def print_topics(F, id_to_wrd, fname, top=9):
    f = open(fname, 'w')
    for i in xrange(F.shape[1]):
        cmd = 'Topic %s: ' % i + ' '.join(map(lambda x: id_to_wrd[x], get_topic(F, i, top))).encode('utf-8') + '\n'
        f.write(cmd)
    f.close()