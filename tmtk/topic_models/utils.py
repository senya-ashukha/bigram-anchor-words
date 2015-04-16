from operator import itemgetter

def get_topic(word_topic, topic, head=10):
    col = list(enumerate(word_topic.T[topic]))
    col = map(itemgetter(0), sorted(col, key=itemgetter(1), reverse=True)[:head])

    return col

def topic(topic, id_to_wrd, head=10, verb=True):
    col = list(enumerate(topic))
    col = map(itemgetter(0), sorted(col, key=itemgetter(1), reverse=True)[:head])
    if verb: print ' '.join(map(lambda x: id_to_wrd[x], col)).encode('utf-8') + '\n'
    return col