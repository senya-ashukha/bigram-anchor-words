from tmtk.topic_models import anchor, plsa

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

from tmtk.collection.transformer_api import TransformerChainApply
from tmtk.collection.transformer import BigramExtractorDocumentsTransform

collection = FullTextCollection(path='./tmtk/corpa/nips.zip', lang='ru').fill()

transformers = TransformerChainApply(transformers=[BigramExtractorDocumentsTransform(do_apply=False)])
collection = transformers.apply(collection)

F, anc = anchor.anchor_model(
    collection,
    k=150,
    wrd_count=collection.num_wrd,
    metrics=[preplexity, coherence, uniq_top_of_topics],
    bi=True)

anchor.print_topics(F, collection.id_to_words, anc, 'en_nips/an+bi.txt')
print 'F_', F.shape

F, T = plsa.plsa_model(
    collection,
    wrd_count=collection.num_wrd,
    metrics=[preplexity, coherence, uniq_top_of_topics],
    num_iter=15, verbose=True, F=F)

plsa.print_topics(F, collection.id_to_words, 'en_nips/an+pl+bi.txt')
print 'F_', F.shape, 'T_', T.shape
