# -*- coding: utf-8 -*-

from tmtk.collection.collection import BagOfWordsCollections

collection_train = BagOfWordsCollections(
    path='/home/oem/Dropbox/projests/Python/tmtk/tmtk/corpa/bag_of_words/nips',
    name='nips_train'
).fill()

