from tmtk.topic_models import anchor, plsa
че это еще за гавно?
from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

from tmtk.collection.transformer_api import TransformerChainApply
from tmtk.collection.transformer import BigramExtractorDocumentsTransform

collection = FullTextCollection(path='./tmtk/corpa/ru_bank_wid_small.zip', lang='ru').fill()

transformers = TransformerChainApply(transformers=[BigramExtractorDocumentsTransform(do_apply=False)])
collection = transformers.apply(collection)

F, anc = anchor.anchor_model(
    collection,
    wrd_count=collection.num_wrd,
    metrics=[preplexity, coherence, uniq_top_of_topics],
    bi=True)

anchor.print_topics(F, collection.id_to_words, anc, 'ru_bank_wid_small/an+bi.txt')

F, T = plsa.plsa_model(
    collection,
    wrd_count=collection.num_wrd,
    metrics=[preplexity, coherence, uniq_top_of_topics],
    num_iter=5, verbose=True, F=F)

plsa.print_topics(F, collection.id_to_words, 'ru_bank_wid_small/an+pl+bi.txt')
