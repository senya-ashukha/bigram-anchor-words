from tmtk.topic_models import plsa

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection, print_bigrams

from tmtk.collection.transformer_api import TransformerChainApply
from tmtk.collection.transformer import BigramExtractorDocumentsTransform

collection = FullTextCollection(path='./tmtk/corpa/ru_bank_wid_small.zip').fill()

transformers = TransformerChainApply(transformers=[BigramExtractorDocumentsTransform()])
collection = transformers.apply(collection)

F, T = plsa.plsa_model(collection.documents_train, collection.documents_test,
                   wrd_count=len(collection.id_to_words),
                   metrics=[preplexity, coherence, uniq_top_of_topics],
                   num_iter=40, verbose=False)

plsa.print_topics(F, collection.id_to_words)
