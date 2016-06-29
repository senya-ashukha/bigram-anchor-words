# About

Bag of words, very poor representation of the text, since a lot of information is lost when it was built. The main objective of the project is the integration of linguistic knowledge in statistical topic model. 
We had modified Anchor Words Topic Model [1] to take into account word collocation, the result is published un conference paper [2]. 

[1] Sanjeev A., Rong G.: A Practical Algorithm for Topic Modeling with Provable Guarantees (NIPS, 2012) 
[2] Ashuha A., Loukachevitch N.: Bigramm Anchor Words Topic Model, Analysis of Images, Social Networks, and Texts (AIST, 2016)

# Experiments 

To repeat published result, you should run flow commands  

```bash
cd bigram-anchor-words
ipython ./examples/{corpus}/{model}.py
```

# Code yourself experiment 

Simple way to repeat experiments is try to understand examples =) I'm really sorry that documentation is absent.  


```python
from tmtk.topic_models import plsa

from tmtk.metrics.metrics import preplexity, coherence, uniq_top_of_topics
from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(path='./tmtk/corpa/20np.zip', lang='ru').fill()

F, T = plsa.plsa_model(
    collection,
    wrd_count=len(collection.id_to_words),
    metrics=[preplexity, coherence, uniq_top_of_topics],
    num_iter=100, verbose=True)

plsa.print_topics(F, collection.id_to_words, 'en_20np/pl.txt')
```