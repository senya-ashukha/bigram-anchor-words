import numpy as np
import random as rnd

def half_random(documents_test):
    estimat_doc, control_doc = [], []
    for d in documents_test:
        est_d, con_d = [], []
        for w, wrdc in d:
            if wrdc > 1:
                est_d.append((w, wrdc/2))
                con_d.append((w, wrdc/2))
            else:
                if rnd.randint(0, 100) % 2:
                    est_d.append((w, wrdc/2))
                else:
                    con_d.append((w, wrdc/2))
        estimat_doc.append(np.array(est_d))
        control_doc.append(np.array(con_d))

    return np.array(estimat_doc), np.array(control_doc)

def estimate_teta(F, documents_test, num_topics=100):
    doc_count = len(documents_test)
    teta = np.zeros((num_topics, doc_count))

    estimate_doc, control_doc = half_random(documents_test)

    for d in xrange(doc_count):
        for w, wrd_count in estimate_doc[d]:
            teta.T[d] += (F[w] * wrd_count)

    for i in xrange(teta.shape[0]):
        for j in xrange(teta.shape[1]):
            if teta[i, j] == 0:
                teta[i, j] = 0.1 ** 5

    for row in teta.T:
        row /= row.sum()
    return teta, control_doc