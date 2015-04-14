import math
import numpy as np

from collections import defaultdict, Counter
from tmtk.metrics.utils import estimate_teta

from tmtk.topic_models.utils import get_topic
from tmtk.collection.collection import bag_of_words

def _preplexity(word_topic, topic_document, documents):
    doc_count = len(topic_document[0])
    lh = 0
    for d in xrange(doc_count):
        for w, ndw in documents[d]:
            Pwd = np.dot(word_topic[w, :], topic_document[:, d])
            lh += ndw * math.log(Pwd if Pwd > 0 else 1e-5)
    return math.e ** (-lh / sum([sum([ndw for _, ndw in documents[d]]) for d in xrange(doc_count)]))

def preplexity(word_topic, train, test):
    bw_train, bw_test = bag_of_words(train), bag_of_words(test)

    topic_document_train, _ = estimate_teta(word_topic, bw_train)
    train_perpl = _preplexity(word_topic, topic_document_train, bw_train)

    topic_document_test, _ = estimate_teta(word_topic, bw_test)
    test_perpl = _preplexity(word_topic, topic_document_test, bw_test)

    return 'preplexity train = %.2f, test = %.2f' % (train_perpl, test_perpl)

def dict_normalize(d):
    denom = sum(d.values())
    for key, value in d.items():
        d[key] = value * 1.0 / denom

    return d

def eval_words_for_probs(word_topic, top):
    words = []
    for t in xrange(word_topic.shape[1]):
        words += get_topic(word_topic, topic=t, head=top)

    return words

def eval_pob_bigrams(docs, num_wrds, words_for_probs, wind_with=10):
    cond_probs = defaultdict(lambda: np.zeros(num_wrds))

    for doc in docs:
        for i in range(len(doc) - wind_with):
            wind = doc[i: i+wind_with]
            for w in wind[:-1]:
                if w in words_for_probs:
                    cond_probs[w][wind[-1]] += 1
                if wind[-1] in words_for_probs:
                    cond_probs[wind[-1]][w] += 1

    for w in cond_probs:
        cond_probs[w] /= sum(cond_probs[w])

    return cond_probs

def eval_pob_ungrams(docs, words_for_probs):
    prob_words = Counter([wrd for doc in docs for wrd in doc])
    prob_words = dict_normalize(prob_words)
    return prob_words

def all_combine(words):
    comb = []
    for i in xrange(len(words)):
        for j in xrange(i, len(words)):
            comb.append((words[i], words[j]))

    return comb

def coherence(word_topic, train, test, top=10, window_with=10):
    words_for_probs = set(eval_words_for_probs(word_topic, top))
    prob_ungrams = eval_pob_ungrams(train, words_for_probs)
    prob_conditn = eval_pob_bigrams(train, word_topic.shape[0], words_for_probs, wind_with=window_with)

    pmi = lambda w1, w2: math.log(prob_conditn[w2][w1] / prob_ungrams[w1]) if prob_conditn[w2][w1] != 0.0 else 0.0
    pmis = []

    for t in xrange(word_topic.shape[1]):
        topic_wrds = get_topic(word_topic, topic=t, head=top)
        pmi_t = [pmi(w1, w2) for w1, w2 in all_combine(topic_wrds) if pmi(w1, w2) != 0.0]
        pmi_t = np.median(pmi_t)
        pmis.append(pmi_t)

    print pmis
    return 'coherence = %.2f' % np.median(pmis)

def uniq_top_of_topics(word_topic, train, test, top=10):
    top_topics = eval_words_for_probs(word_topic, top)

    mesure = len(set(top_topics)) * 1.0 / len(top_topics)

    return 'uniq_top_of_topics = %.2f' % mesure