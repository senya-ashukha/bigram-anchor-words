import zipfile
import numpy as np
from operator import itemgetter

from string import split
from itertools import imap

from collections import Counter
from tmtk.utils.iter import grouper

class Collection():
    def __init__(self, path):
        self.path = path

        self.documents_train = list()
        self.documents_test  = list()

        self.id_to_words = dict()
        self.words_to_id = dict()

    def fill(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

class FullTextCollection(Collection):
    def fill(self):
        zf = zipfile.ZipFile(self.path)

        if not zf.namelist() == ['test.txt', 'train.txt', 'vocab.txt']:
            raise Exception('Collection arch must be contain only this files: test.txt, train.txt, vocab.txt')

        for text_id, doc in grouper(zf.open('train.txt'), 2):
            self.documents_train.append(map(int, doc.decode('utf-8').strip().split()))

        for text_id, doc in grouper(zf.open('test.txt'), 2):
            self.documents_test.append(map(int, doc.decode('utf-8').strip().split()))

        self.id_to_words = dict(filter(len, imap(split, zf.open('vocab.txt').read().decode('utf8').split('\n'))))
        self.id_to_words = dict(zip(map(int, self.id_to_words.keys()), self.id_to_words.values()))
        self.words_to_id = dict(imap(lambda x: (x[1], x[0]), self.id_to_words.iteritems()))

        self.num_wrd = len(self.id_to_words)

        print 'Read %s num doc train = %s test = %s num wrd %s' % \
              (self.path, len(self.documents_train), len(self.documents_test), self.num_wrd)

        return self

def bag_of_words(documents):
    bw_collection = [Counter([wrd for wrd in doc]).items() for doc in documents]
    return np.array(bw_collection)

def print_bigrams(collection, top=10):
    idx = sorted(collection.bigrams.items(), key=itemgetter(1), reverse=True)[:top]
    for i, c in idx: print collection.id_to_words[collection.words_to_id[i]], c
