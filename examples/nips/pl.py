from tmtk.topic_models import plsa

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(path='./tmtk/corpa/nips.zip', lang='ru').fill()

F, T = plsa.plsa_model(
    collection,
    wrd_count=len(collection.id_to_words),
    metrics=[preplexity, coherence, uniq_top_of_topics],
    num_iter=40, verbose=True)

plsa.print_topics(F, collection.id_to_words, 'en_nips/pl.txt')
print 'F_', F.shape, 'T_', T.shape
