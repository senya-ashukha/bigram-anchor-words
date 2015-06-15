from tmtk.topic_models import anchor, plsa

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(path='./tmtk/corpa/ru_bank_wid_small.zip').fill()

F, anc = anchor.anchor_model(
    collection,
    wrd_count=len(collection.id_to_words),
    metrics=[preplexity, coherence, uniq_top_of_topics])

anchor.print_topics(F, collection.id_to_words, anc, 'ru_bank_wid_small/an+bi.txt')

F, T = plsa.plsa_model(
    collection,
    wrd_count=len(collection.id_to_words),
    metrics=[preplexity, coherence, uniq_top_of_topics],
    num_iter=40, verbose=False, F=F)

plsa.print_topics(F, collection.id_to_words, 'ru_bank_wid_small/an+pl+bi.txt')