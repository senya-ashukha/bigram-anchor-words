# -*- coding: utf-8 -*-

from tmtk.utils import pickle

collection = pickle.load('text_collection')
for i, doc in enumerate(collection.yield_str_documents()):
    print 'doc %i: %s' % (i, doc)

