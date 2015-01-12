from abc import abstractmethod

class Collection():
    """
    in fill your must init:
        raw_data - variable for your row data
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

        self.documents = None
        self.words_map = None
        self.wc_mtx = None

    def __iter__(self):
        for document in self.documents:
            yield document

    def __str__(self):
        s = ''
        for document in self.documents:
            for sent in document:
                s += u' '.join(sent) + '. '
            s += '\n'

        return s.encode('utf-8')

    def word_iter_generate(self):
        for document in self.documents:
            for sent in document:
                for word in sent:
                    yield word

    @classmethod
    def document_word_iter_generate(cls, document):
        for sent in document:
            for word in sent:
                yield word

    def voc(self, id_s=False):
        if not self.words_map:
            raise Exception('run mem optimize method.')

        return self.words_map.values() if id_s else self.words_map.keys()

class NLTKCollections(Collection):
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

class FileCollections(Collection):
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