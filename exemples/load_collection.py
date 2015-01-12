import os

from tmtk.utils import pickle
from itertools import izip
from collections import defaultdict

base_path = '/Users/ars/Dropbox/Projests/Python/PycharmProjects/tmtk/ad_hock/cowords'

collections = pickle.load('wiki_ru_featured_articles')

print 'loaded'
print len(collections.documents)
print len(collections.doc_cat)

document_cat_pairs = izip(collections.documents, collections.doc_cat)

for doc_id, (_, cat) in enumerate(document_cat_pairs):
    cat_path = os.path.join(base_path, cat)

    if not os.path.exists(cat_path):
        os.mkdir(cat_path)

    f = open(os.path.join(cat_path, str(doc_id) + '.txt'), 'w')

    for bigram, (messure, counts) in collections.bigrams[doc_id].items():
        f.write('%s %s %s %s\n' % (bigram[0].encode('utf-8'), bigram[1].encode('utf-8'), messure, counts))

    f.close()

cur_cat = ''
document_cat_pairs = izip(collections.documents, collections.doc_cat)

for doc_id, (_, cat) in enumerate(document_cat_pairs):
    if cur_cat != cat:
        if cur_cat != '':
            f = open(os.path.join(cat_path, 'intersection' + '.txt'), 'w')

            for bigram, count in filter(lambda x: x[1] > 1, reversed(sorted(d.items(), key=lambda x: x[1]))):
                f.write('%s %s %s\n' % (bigram[0].encode('utf-8'), bigram[1].encode('utf-8'), count))

            f.close()

        cur_cat  = cat
        cat_path = os.path.join(base_path, cat)
        d = defaultdict(lambda: 0)

    for bigram, (_, _) in collections.bigrams[doc_id].items():
        d[bigram] += 1

f = open(os.path.join(cat_path, 'intersection' + '.txt'), 'w')

for bigram, count in filter(lambda x: x[1] > 1, reversed(sorted(d.items(), key=lambda x: x[1]))):
    f.write('%s %s %s\n' % (bigram[0].encode('utf-8'), bigram[1].encode('utf-8'), count))

f.close()