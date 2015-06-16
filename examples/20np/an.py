from tmtk.topic_models import anchor

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(path='./tmtk/corpa/20np.zip', lang='en').fill()

F, anc = anchor.anchor_model(
    collection, wrd_count=len(collection.id_to_words), metrics=[preplexity, coherence, uniq_top_of_topics], k=50)
anchor.print_topics(F, collection.id_to_words, anc, '20np/an.txt')