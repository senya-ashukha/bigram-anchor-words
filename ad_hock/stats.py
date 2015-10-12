import collections

with open('./results/ru_bank_wid_small/z_bigrams_stats.txt', 'w') as bi:
    for (w1, w2), count in reversed(sorted(collection.bigrams.items(), key=lambda x: x[1])):
        a = '%s %s\t%s\n' % (collection.id_to_words[w1], collection.id_to_words[w2], count)
        bi.write(a.encode('utf8'))

with open('./results/ru_bank_wid_small/z_unigrams_stats.txt', 'w') as bi:
    wrds = [wrd for doc in collection.documents_train for wrd in doc] + [wrd for doc in collection.documents_test for wrd in doc]
    wrds = collections.Counter(wrds)
    for w, count in reversed(sorted(wrds.items(), key=lambda x: x[1])):
        a = '%s\t%s\n' % (collection.id_to_words[w], count)
        bi.write(a.encode('utf8'))