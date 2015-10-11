from tmtk.topic_models import anchor

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

from tmtk.utils.logger import get_logger
import gc

logger = get_logger()

collection = FullTextCollection(path='./tmtk/corpa/20np.zip', lang='ru').fill()

F, anc = anchor.anchor_model(
    collection, wrd_count=len(collection.id_to_words), k=300, metrics=[preplexity, coherence, uniq_top_of_topics])

anchor.print_topics(F, collection.id_to_words, anc, 'en_20np/an.txt')

print gc.collect()
