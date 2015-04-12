from tmtk.topic_models import anchor

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

from tmtk.collection.transformer_api import TransformerChainApply
from tmtk.collection.transformer import BigramExtractorDocumentsTransform

collection = FullTextCollection(path='./tmtk/corpa/ru_bank_wid_small.zip').fill()

transformers = TransformerChainApply(transformers=[BigramExtractorDocumentsTransform()])
collection = transformers.apply(collection)

F, anc = anchor.anchor_model(
    collection, wrd_count=len(collection.id_to_words), metrics=[preplexity, coherence, uniq_top_of_topics])


anchor.print_topics(F, collection.id_to_words, anc)