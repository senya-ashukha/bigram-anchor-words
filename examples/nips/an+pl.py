from tmtk.topic_models import anchor, plsa

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(path='./tmtk/corpa/nips.zip', lang='ru').fill()

F, anc = anchor.anchor_model(
    collection,
    wrd_count=len(collection.id_to_words),
    k=150,
    metrics=[preplexity, coherence, uniq_top_of_topics])

anchor.print_topics(F, collection.id_to_words, anc, 'en_20np/an.txt')
print 'F_', F.shape

F, T = plsa.plsa_model(
    collection,
    wrd_count=len(collection.id_to_words),
    metrics=[preplexity, coherence, uniq_top_of_topics],
    num_iter=15, verbose=True, F=F)

plsa.print_topics(F, collection.id_to_words, 'en_nips/an+pl.txt')
print 'F_', F.shape, 'T_', T.shape