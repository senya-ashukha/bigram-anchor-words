import nltk, os
from collections import Counter
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))

porter = nltk.PorterStemmer()

en = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
digit, space = '0123456789', ' \t\n'
good_symbol = en + en.lower() + digit + space


def filter_word(word):
    return u''.join(filter(lambda char: char in good_symbol, list(word)))


def filter_bw(text):
    text = text.split()
    text = filter(lambda x: len(x) > 2, text)
    text = filter(lambda x: not x.isdigit(), text)
    return ' '.join(text)


def stem(text):
    text = text.split()
    text = map(porter.stem, text)
    return ' '.join(text)


# cd train

texts = []

for f in os.listdir('.'):
    try:
        a = stem(filter_bw(filter_word(' '.join(open(f).read().split()).replace('\n', ' ')).lower()))
        texts.append(a)
    except:
        print f

vocab = Counter([w for t in texts for w in t.split()])
vocab = filter(lambda x: x not in sw, dict(filter(lambda x: x[1] > 2, vocab.items())).keys())

wrd2id = dict([(w, i) for i, w in enumerate(vocab)])

id_texts = []

for t in texts:
    txt = []
    for w in t.split():
        if w in wrd2id:
            txt.append(wrd2id[w])
    id_texts.append(txt)

with open('../train.txt', 'w') as train_f:
    for i, t in enumerate(filter(len, id_texts)):
        t = ' '.join(map(str, t))
        train_f.write('%s\n%s\n' % (i, t))



# cd ../test

texts = []

for f in os.listdir('.'):
    a = stem(filter_bw(filter_word(' '.join(open(f).read().split()).replace('\n', ' ')).lower()))
    texts.append(a)

id_texts = []

for t in texts:
    txt = []
    for w in t.split():
        if w in wrd2id:
            txt.append(wrd2id[w])
    id_texts.append(txt)

with open('../test.txt', 'w') as train_f:
    for i, t in enumerate(filter(len, id_texts)):
        t = ' '.join(map(str, t))
        train_f.write('%s\n%s\n' % (i, t))

with open('../vocab.txt', 'w') as train_f:
    for w, i in sorted(wrd2id.items(), key=lambda x: x[1]):
        train_f.write('%s %s\n' % (i, w))
