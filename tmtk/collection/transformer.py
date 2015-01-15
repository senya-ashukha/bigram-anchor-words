# -*- coding: utf-8 -*-

from multiprocessing import Pool

from collections import Counter, defaultdict
from nltk.util import ngrams
from scipy.sparse import dok_matrix

from tmtk.utils.iter import grouper, all_pairs
from tmtk.utils.dict import dicts_sum
from tmtk.utils.lingvo import doc_normalizer, doc_stop_word_remove
from tmtk.collection.collection import Collection

class TransformerApplyer():
    def __init__(self, transformers, verbose=False):
        self.transformers = transformers
        self.verbose = verbose

    def apply(self, collections):
        for transformer in self.transformers:
            if self.verbose: print 'Train:\t\t' + str(transformer.__class__)
            transformer.train(collections)

            if self.verbose: print 'Apply:\t\t' + str(transformer.__class__)
            transformer.apply(collections)

            if self.verbose: print 'Finished:\t' + str(transformer.__class__) + '\n'

class Transformer():
    def train(self, collection):
        raise NotImplementedError

    def apply(self, collection):
        raise NotImplementedError

class MultiThreadMapTransformApplyer():
    def __init__(self, core=1):
        self.core = core

    def train(self, collection):
        raise NotImplementedError('define self,map')

    def apply(self, collection):
        collection.documents = Pool(self.core).map(self.map, collection)

class PunctuationRemoverTransform(Transformer):
    def train(self, collection):
        pass

    def apply(self, collection):
        ru, en, digit, space = [u'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ', u'ABCDEFGHIJKLMNOPQRSTUVWXYZ', u'0123456789',
                                u' \t\n']
        good_symbol = ru + ru.lower() + en + en.lower() + digit + space

        def filter_word(word):
            return u''.join(filter(lambda char: char in good_symbol, list(word)))

        collection.documents = [[map(filter_word, sent) for sent in document] for document in collection]

class LoweCaseTransform(Transformer):
    def train(self, collection):
        pass

    def apply(self, collection):
        collection.documents = [[[word.lower() for word in sent] for sent in document] for document in collection]

class WordNormalizerTransform(MultiThreadMapTransformApplyer):
    def train(self, collection):
        self.map = doc_normalizer

class StopWordsRemoverTransform(MultiThreadMapTransformApplyer):
    def train(self, collection):
        self.map = doc_stop_word_remove

class BigramExtractorDocumentsTransform(Transformer):
    def __init__(self, window_width=5, sigma=0.5, min_occur=1, min_word_len=3):
        self.window_width = window_width
        self.sigma = sigma
        self.min_occur = min_occur
        self.min_word_len = min_word_len

    def train(self, collection):
        collection.bigrams = dict()
        collocation_measure = lambda coloc: counts_neighbors[coloc] * 1.0 / counts_windows[coloc]

        for i, document in enumerate(collection):
            groups_words = grouper(Collection.document_word_iter_generate(document), 5)
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

            len_check = lambda word: word and len(word) >= self.min_word_len
            buf = filter(lambda bigram: len_check(bigram[0][0]) and len_check(bigram[0][1]), buf)
            collection.bigrams[i] = dict(buf)

    def apply(self, collection):
        def bigram_replace((i, document)):
            new_document = []
            for sent in document:
                new_sent = []
                for bigram in ngrams(sent + [''], 2):
                    new_sent.append('_'.join(bigram) if bigram in collection.bigrams[i] else bigram[0])
                new_document.append(new_sent)

            return new_document

        collection.documents = map(bigram_replace, enumerate(collection))

class MemOptimizeTransform(Transformer):
    def train(self, collection):
        pass

    def apply(self, collection):
        if not collection.documents:
            raise Exception('self.documents is None, fill your collection.')

        collection.words_map, id_s = {}, 0

        for document in collection.documents:
            for sent in document:
                for word in sent:
                    if word not in collection.words_map:
                        collection.words_map[word] = id_s
                        id_s += 1

class CreateWCMatrixTransform(Transformer):
    def train(self, collection):
        pass

    def apply(self, collection):
        mtx_class=dok_matrix
        if not collection.words_map:
            raise Exception('run mem optimize method.')

        collection.wc_mtx = mtx_class((len(collection.documents), len(collection.words_map)))

        for num_document, document in enumerate(collection.documents):
            for sent in document:
                for word in sent:
                    if word not in document:
                        collection.wc_mtx[num_document, collection.words_map[word]] += 1

class ShortSentRemoverTransform(Transformer):
    def __init__(self, min_len=1):
        self.min_len = min_len

    def train(self, collection):
        pass

    def apply(self, collection):
        collection.documents = [[sent for sent in document if len(sent) >= self.min_len] for document in collection]


