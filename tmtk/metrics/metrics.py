import math

import numpy as np

from tmtk.metrics.utils import estimate_teta

def _preplexity(word_topic, topic_document, documents):
    doc_count = len(topic_document[0])
    lh = 0
    for d in xrange(doc_count):
        for w, ndw in documents[d]:
            Pwd = np.dot(word_topic[w, :], topic_document[:, d])
            lh += ndw * math.log(Pwd if Pwd > 0 else 1e-5)
    return math.e ** (-lh / sum([
        sum([ndw for _, ndw in documents[d]]) for d in xrange(doc_count)]))

def preplexity(word_topic, documents_train, documents_test):
    topic_document_train, _ = estimate_teta(word_topic, documents_train)
    train_perpl = _preplexity(word_topic, topic_document_train, documents_train)

    topic_document_test, _ = estimate_teta(word_topic, documents_test)
    test_perpl = _preplexity(word_topic, topic_document_test, documents_test)

    return 'preplexity train = %.2f, test = %.2f' % (train_perpl, test_perpl)

