from tmtk.topic_models import plsa

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

from tmtk.collection.transformer_api import TransformerChainApply
from tmtk.collection.transformer import BigramExtractorDocumentsTransform


collection_train = FullTextCollection(
    path='/home/oem/Dropbox/projests/Python/tmtk/tmtk/corpa/full_text/wiki_ru_article',
    name='wiki_ru_article_train'
).fill()

collection_test = FullTextCollection(
    path='/home/oem/Dropbox/projests/Python/tmtk/tmtk/corpa/full_text/wiki_ru_article',
    name='wiki_ru_article_test'
).fill()

transformers = [BigramExtractorDocumentsTransform()]
applyer = TransformerChainApply(transformers=transformers)

collection_train = applyer.apply(collection_train)
collection_test.bigrams = collection_train.bigrams
collection_test = transformers[0].apply(collection_test)

F, T = plsa.plsa_model(collection_train, collection_test,
                       wrd_count=len(collection_train.id_to_words),
                       metrics=[preplexity, coherence, uniq_top_of_topics],
                       num_iter=40, verbose=False)

plsa.print_topics(F, collection_train.id_to_words)
