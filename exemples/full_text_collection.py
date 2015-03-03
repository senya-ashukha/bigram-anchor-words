# -*- coding: utf-8 -*-

from tmtk.collection.collection import FullTextCollection

collection = FullTextCollection(
    collection_name='wiki_ru_article',
    path='/home/oem/Dropbox/projests/Python/tmtk/tmtk/corpa/full_text/wiki_ru_article',
    name='wiki_ru_article'
).fill()
collection.save()