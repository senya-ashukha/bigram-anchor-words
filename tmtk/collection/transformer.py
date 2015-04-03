# -*- coding: utf-8 -*-

from collections import Counter, defaultdict

from nltk.util import ngrams
#from scipy.sparse import dok_matrix

from tmtk.collection.transformer_api import *
from tmtk.utils.iter import grouper, all_pairs
from tmtk.utils.dict import dicts_sum
from tmtk.utils.lingvo import doc_normalizer, doc_stop_word_remove

from operator import itemgetter


class PunctuationRemoverTransform(Transformer):
    def train(self, collection):
        pass

    def apply(self, collection):
        ru, en = u'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', u'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        digit, space = u'0123456789', u' \t\n'
        good_symbol = ru + ru.lower() + en + en.lower() + digit + space

        def filter_word(word):
            return u''.join(filter(lambda char: char in good_symbol, list(word)))

        collection.documents = [[map(filter_word, sent) for sent in document] for document in collection.documents]

class LoweCaseTransform(Transformer):
    def train(self, collection):
        pass

    def apply(self, collection):
        collection.documents = [[[word.lower() for word in sent] for sent in document]
                                for document in collection.documents]

class EmptyWordRemoverTransform(Transformer):
    def train(self, collection):
        pass

    def apply(self, collection):
        collection.documents = [[filter(len, sent) for sent in document] for document in collection.documents]

class WordNormalizerTransform(MultiThreadTransformer):
    def train(self, collection):
        self.map = doc_normalizer

class StopWordsRemoverTransform(MultiThreadTransformer):
    def train(self, collection):
        self.map = doc_stop_word_remove

class BigramExtractorDocumentsTransform(Transformer):
    def __init__(self, window_width=5, sigma=0.5, min_occur=1, min_word_len=3, top=3000):
        Transformer.__init__(self)
        self.window_width = window_width
        self.sigma = sigma
        self.min_occur = min_occur
        self.min_word_len = min_word_len
        self.top = top

    def train(self, collection):
        bigrams = dict()
        collocation_measure = lambda coloc: counts_neighbors[coloc] * 1.0 / counts_windows[coloc]

        for i, document in enumerate(collection.documents):
            groups_words = grouper([wrd for sent in document for wrd in sent], 5)
            counts_neighbors = defaultdict(lambda: 0)
            counts_windows = defaultdict(lambda: 0)

            for group in groups_words:
                count_neighbor = Counter(ngrams(group, 2))
                count_window = Counter(all_pairs(group))

                dicts_sum(counts_neighbors, count_neighbor)
                dicts_sum(counts_windows, count_window)

            buf = []
            for item in counts_neighbors:
                if collocation_measure(item) > self.sigma and counts_neighbors[item] > self.min_occur and item[0] != item[1]:
                    buf.append((item, (collocation_measure(item), counts_neighbors[item])))

            bigrams[i] = dict(buf)

        collection.bigrams = []
        for i in xrange(len(collection.documents)):
            collection.bigrams += bigrams[i].keys()
        collection.bigrams = set(collection.bigrams)

        documents = [wrd for document in collection.documents for sent in document for wrd in sent]
        bigrams = filter(lambda bigr: bigr in collection.bigrams, ngrams(documents, 2))
        collection.bigrams = dict(sorted(Counter(bigrams).items(), key=itemgetter(1), reverse=True)[:self.top])

    def apply(self, collection):
        max_v = max(collection.words_to_id.values()) + 1

        for bigram in collection.bigrams.keys():
            collection.words_to_id[bigram] = max_v
            collection.id_to_words[max_v] = collection.id_to_words[bigram[0]] + '_' + collection.id_to_words[bigram[1]]
            max_v += 1

        new_documents = []
        for document in collection.documents:
            new_document = []
            for sent in document:
                bigrams = ngrams(sent, 2)
                f = 1
                new_sent = []
                for bigram in bigrams:
                    if f:
                        if bigram in collection.bigrams:
                            new_sent.append(collection.words_to_id[bigram])
                            f = 0
                        else:
                            new_sent.append(bigram[0])
                    else:
                        f = 1
                new_document.append(new_sent)
            new_documents.append(new_document)
        collection.documents = new_documents

        return collection

class ShortSentRemoverTransform(Transformer):
    def __init__(self, min_len=1):
        self.min_len = min_len

    def train(self, collection):
        pass

    def apply(self, collection):
        collection.documents = [[sent for sent in document if len(sent) >= self.min_len]
                                for document in collection.documents]

class TrashFilterTransform(Transformer):
    def __init__(self, min_len=2, min_occ=30):
        Transformer.__init__(self)
        self.min_len = min_len
        self.min_occ = min_occ
        self.no_trash_wrds = None
        self.map = lambda doc: [filter(lambda wrd: wrd in self.no_trash_wrds, sent) for sent in doc]

    def train(self, collection):
        coll = Counter([wrd for doc in collection.documents for sent in doc for wrd in sent])

        self.no_trash_wrds = dict(
            filter(lambda x: x[1] > self.min_occ and len(collection.id_to_words[x[0]]) > self.min_len, coll.items())
        )

        print len(self.no_trash_wrds)

    def apply(self, collection):
        collection.documents = map(self.map, collection.documents)