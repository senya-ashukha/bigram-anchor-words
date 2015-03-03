from multiprocessing import Pool

class Transformer():
    def __init__(self):
        pass

    def train(self, collection):
        raise NotImplementedError

    def apply(self, collection):
        raise NotImplementedError

class MultiThreadTransformer():
    def __init__(self, core=1):
        self.core = core
        self.map = None

    def train(self, collection):
        raise NotImplementedError('define self,map')

    def apply(self, collection):
        collection.documents = Pool(self.core).map(self.map, collection.documents)

class TransformerChainApply():
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