import os
from tmtk.utils.iter import grouper

class Collection():
    def __init__(self, collection_name, path, name):
        self.collection_name = collection_name

        self.path = path
        self.name = name

        self.documents, self.topics = [], []
        self.topics_id, self.words_id = {}, {}

    def fill(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

class BagOfWordsCollections(Collection):
    pass

class FullTextCollection(Collection):
    def fill(self):
        col_nam, voc_name, top_name = self.name + '.txt', self.name + '.voc.txt', self.name + '.top.txt'

        col_nam = os.path.join(self.path, col_nam)
        voc_name = os.path.join(self.path, voc_name)
        top_name = os.path.join(self.path, top_name)

        self.documents, self.topics = [], []

        for id, topic, doc in grouper(open(col_nam), 3):
            topic, doc = topic.decode('utf-8').strip(), doc.decode('utf-8').strip()
            self.documents.append(map(lambda x: x.split(), doc.split('#')))
            self.topics.append(topic)

        self.id_to_topics = dict(enumerate(map(lambda x: x.decode('utf-8').strip(), open(top_name).readlines())))
        self.topics_to_id = dict(map(lambda x: (x[1], x[0]), self.id_to_topics.iteritems()))

        self.id_to_words = dict(enumerate(map(lambda x: x.decode('utf-8').strip(), open(voc_name).readlines())))
        self.words_to_id = dict(map(lambda x: (x[1], x[0]), self.id_to_words.iteritems()))

        return self

    def save(self):
        col_nam, voc_name, top_name = self.name + '.txt', self.name + '.voc.txt', self.name + '.top.txt'
        col_nam = os.path.join(self.path, col_nam)
        voc_name = os.path.join(self.path, voc_name)
        top_name = os.path.join(self.path, top_name)

        str_doc = lambda doc: '#'.join([' '.join(map(str, sent)) for sent in doc])

        f_out = open(col_nam, 'w')
        for id, topic, doc in zip(xrange(len(self.documents)), self.topics, self.documents):
            line = '%s\n%s\n%s\n' % (id, topic, str_doc(doc))
            f_out.write(line.encode('utf-8'))

        f_out = open(voc_name, 'w')
        words = map(lambda x: x[1], sorted(self.id_to_words.iteritems(), key=lambda x: x[0]))
        words = '\n'.join(words)
        f_out.write(words.encode('utf-8'))

        f_out = open(top_name, 'w')
        words = map(lambda x: x[1], sorted(self.id_to_topics.iteritems(), key=lambda x: x[0]))
        words = '\n'.join(words)
        f_out.write(words.encode('utf-8'))
