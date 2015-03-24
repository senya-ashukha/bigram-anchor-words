import math

import numpy as np

from collections import defaultdict
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

def preplexity(word_topic, documents_train, documents_test, **args):
    topic_document_train, _ = estimate_teta(word_topic, documents_train)
    train_perpl = _preplexity(word_topic, topic_document_train, documents_train)

    topic_document_test, _ = estimate_teta(word_topic, documents_test)
    test_perpl = _preplexity(word_topic, topic_document_test, documents_test)

    return 'preplexity train = %.2f, test = %.2f' % (train_perpl, test_perpl)

def coherence(word_topic, documents_train, documents_test, top=10):
    def all_combination(words):
        words, comb = list(sorted(set(words))), []

        for i in xrange(len(words)):
            for j in xrange(i, len(words)):
                comb.append((words[i], words[j]))

        return comb

    def ngrams_eval(w_for_ngrams):
        w_for_ngrams = set(w_for_ngrams)
        bigrams, unigrams = defaultdict(lambda: 0), defaultdict(lambda: 0)

        return bigrams, unigrams

    def PMI(w1, w2):
        return math.log((num_doc*bigrams[w1, w2])*1.0/(unigrams[w1]*unigrams[w2]))

    num_wrd, num_top, num_doc = word_topic.shape[0], word_topic.shape[1], word_topic.shape[1]

    w_for_ngrams = []
    for t in xrange(num_top):
        w = topic(word_topic, topic=t, head=top)
        w_for_ngrams += w

    bigrams, unigrams = ngrams_eval(w_for_ngrams)

    coher = []
    for t in xrange(num_top):
        w = topic(word_topic, topic=t, head=top)
        PMIt = 2/(top*(top-1)) * sum([sum([PMI(w[i], w[j]) for j in xrange(i, top)]) for i in xrange(top-1)])
        coher.append(PMIt)

    return sum(coher)/len(coher)