from operator import itemgetter

def get_topic(word_topic, topic, head=10):
    col = list(enumerate(word_topic.T[topic]))
    col = map(itemgetter(0), sorted(col, key=itemgetter(1), reverse=True)[:head])

    return col