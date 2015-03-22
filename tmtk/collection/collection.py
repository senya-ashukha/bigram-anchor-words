import os
import numpy as np

from tmtk.utils.iter import grouper

class Collection():
    def __init__(self, path, name):
        self.path = path
        self.name = name

        self.documents, self.topics = [], []
        self.topics_id, self.words_id = {}, {}

    def fill(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

class BagOfWordsCollections(Collection):
    def __init__(self, path, name):
        Collection.__init__(self, path, name)
        self.col_nam = os.path.join(self.path, self.name + '.txt')
        self.voc_name = os.path.join(self.path, '_'.join(self.name.split('_')[:-1]) + '.voc.txt')
        self.top_name = os.path.join(self.path, self.name + '.top.txt')

    def fill(self):
        self.documents = []
        self.topics = None

        cur_doc, cur_doc_id = [], 0

        col = open(self.col_nam)

        self.num_wrd = int(col.readline())

        for line in col:
            doc_id, wrd, wrd_count = map(int, line.split())
            if doc_id != cur_doc_id:
                self.documents.append(cur_doc)
                cur_doc = []
                cur_doc_id = doc_id
            cur_doc.append((wrd-1, wrd_count))

        if cur_doc:
            self.documents.append(cur_doc)
        self.documents = np.array(self.documents)

        self.id_to_words = dict(
            enumerate(map(lambda x: x.decode('utf-8').strip(), open(self.voc_name).readlines())))
        self.words_to_id = dict(map(lambda x: (x[1], x[0]), self.id_to_words.iteritems()))

        print 'load', self.col_nam.split('/')[-1], 'num_docs', len(self.documents), 'num_wrds', self.num_wrd
        return self

    def save(self):
        f_out = open(self.col_nam, 'w')
        for id, doc in enumerate(self.documents):
            for wrd, count in doc:
                f_out.write((u'%s %s %s\n' % (id, wrd, count)).encode('utf-8'))

        self.id_to_words = dict(
            enumerate(map(lambda x: x.decode('utf-8').strip(), open(self.voc_name).readlines())))
        self.words_to_id = dict(map(lambda x: (x[1], x[0]), self.id_to_words.iteritems()))

class FullTextCollection(Collection):
    def __init__(self, path, name):
        Collection.__init__(self, path, name)
        self.col_nam = os.path.join(self.path, self.name + '.txt')
        self.voc_name = os.path.join(self.path, '_'.join(self.name.split('_')[:-1]) + '.voc.txt')
        self.top_name = os.path.join(self.path, self.name + '.top.txt')

    def fill(self):
        self.documents, self.topics = [], []

        for id, topic, doc in grouper(open(self.col_nam), 3):
            topic, doc = topic.decode('utf-8').strip(), doc.decode('utf-8').strip()
            self.documents.append(map(lambda x: map(int, x.split()), doc.split('#')))
            self.topics.append(topic)

        self.id_to_topics = dict(
            enumerate(map(lambda x: x.decode('utf-8').strip(), open(self.top_name).readlines())))
        self.topics_to_id = dict(map(lambda x: (x[1], x[0]), self.id_to_topics.iteritems()))

        self.id_to_words = dict(
            enumerate(map(lambda x: x.decode('utf-8').strip(), open(self.voc_name).readlines())))
        self.words_to_id = dict(map(lambda x: (x[1], x[0]), self.id_to_words.iteritems()))

        return self

    def save(self):
        str_doc = lambda doc: '#'.join([' '.join(map(str, sent)) for sent in doc])

        f_out = open(self.col_nam, 'w')
        for id, topic, doc in zip(xrange(len(self.documents)), self.topics, self.documents):
            line = '%s\n%s\n%s\n' % (id, topic, str_doc(doc))
            f_out.write(line.encode('utf-8'))

        f_out = open(self.voc_name, 'w')
        words = map(lambda x: x[1], sorted(self.id_to_words.iteritems(), key=lambda x: x[0]))
        words = '\n'.join(words)
        f_out.write(words.encode('utf-8'))

        f_out = open(self.top_name, 'w')
        words = map(lambda x: x[1], sorted(self.id_to_topics.iteritems(), key=lambda x: x[0]))
        words = '\n'.join(words)
        f_out.write(words.encode('utf-8'))
