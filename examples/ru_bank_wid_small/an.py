from tmtk.topic_models import anchor

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(path='./tmtk/corpa/ru_bank_wid_small.zip', lang='ru').fill()

F, anc = anchor.anchor_model(
    collection, wrd_count=len(collection.id_to_words), k=400, metrics=[preplexity, coherence, uniq_top_of_topics])

anchor.print_topics(F, collection.id_to_words, anc, 'ru_bank_wid_small/an.txt')