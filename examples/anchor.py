from tmtk.topic_models import anchor

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(path='./tmtk/corpa/ru_bank_wid_small.zip').fill()


F, anc = anchor.anchor_model(collection.documents_train, collection.documents_test,
                             wrd_count=len(collection.id_to_words),
                             metrics=[preplexity, coherence, uniq_top_of_topics])


anchor.print_topics(F, collection.id_to_words, anc)