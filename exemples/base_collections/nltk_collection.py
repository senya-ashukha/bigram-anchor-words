# -*- coding: utf-8 -*-

from nltk.corpus import *

from tmtk.collection import transformer, collection

from tmtk.utils import pickle

wiki_ru_featured_articles = LazyCorpusLoader(
    'wiki_ru_featured_articles', CategorizedPlaintextCorpusReader, r'[a-z][a-z][a-z][0-9][0-9][0-9]',
    cat_file='tcats.txt', encoding='utf-8'
)

collections = collection.NLTKCollections(collection_name='wiki_ru_featured_articles', raw_data=wiki_ru_featured_articles)
collections.fill()

transformers = transformer.TransformerApplyer([
    transformer.PunctuationRemoverTransform(),
    transformer.LoweCaseTransform(),
    transformer.WordNormalizerTransform(core=3),
    transformer.StopWordsRemoverTransform(core=3),
    transformer.ShortSentRemoverTransform(),
    transformer.BigramExtractorDocumentsTransform(),

    transformer.MemOptimizeTransform(),
    transformer.CreateWCMatrixTransform()

], verbose=True)

transformers.apply(collections)
pickle.dump(collections)

print 'finish'