import numpy as np
from abc import abstractmethod

class Collection():
    """
    in fill your must init:
        raw_data - variable for your row documents_collection
        documents - list with list(documents) of list(sentences) of words
        collection_name - your collection name

    automatic init:
        words_map - map with {word: int_word_id}
        wc_mtx    - sparse word count matrix
    """

    @abstractmethod
    def fill(self):
        pass

    def __init__(self, collection_name, raw_data=None):
        self.collection_name = collection_name
        self.raw_data = raw_data

        self.documents_train = None
        self.documents_test  = None

    def iter_documents_train(self):
        for document in self.documents_train:
            yield document

    def iter_documents_test(self):
        for document in self.documents_test:
            yield document

    def yield_str_documents(self):
        for document in self.documents:
            yield ' '.join([w for sent in document for w in sent])

    def bag_of_words(self):
        raise NotImplementedError()

class FullTextCollection(Collection):
    """
        raw_date: is a nltk corpus.
    """

    def fill(self):
        self.documents, self.doc_cat = [], []
        for doc_name in self.raw_data.fileids():
            sentences = self.raw_data.sents(doc_name)
            self.documents.append(sentences)
            category = self.raw_data.categories(doc_name)[0]
            self.doc_cat.append(category)
        self.raw_data = []

    def bag_of_words(self):
        def wc_mtx(docs):
            return np.array([123])
        vocab = None
        return wc_mtx(self.documents_train), wc_mtx(self.documents_test), vocab

class BagOfWordsCollections(Collection):
    """
         raw_date: is a list of string(paths to documents files).
    """

    def fill(self):
        self.documents = []
        for doc_name in self.raw_data:
            document = u' '.join(open(doc_name).readlines())
            sentences = document.split(u'.')
            sentences = [sentence.split() for sentence in sentences]
            self.documents.append(sentences)

    def bag_of_words(self):
        def wc_mtx(docs):
            return np.array([123])
        vocab = None
        return wc_mtx(self.documents_train), wc_mtx(self.documents_test), vocab

class StringCollections(Collection):
    """
        raw_date: is a list of string(document).
    """

    def fill(self):
        self.documents = []
        for doc in self.raw_data:
            sentences = filter(None, doc.split('.'))
            sentences = [sentence.split() for sentence in sentences]
            self.documents.append(sentences)

    def bag_of_words(self):
        def wc_mtx(docs):
            return np.array([123])
        vocab = None
        return wc_mtx(self.documents_train), wc_mtx(self.documents_test), vocab
