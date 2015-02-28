# -*- coding: utf-8 -*-

import glob
import os

from defau.defaults import abbreviation

base_dump_path = '/Users/ars/Dropbox/Projests/Python/PycharmProjects/tmtk/ad_hock/download_wiki/documents_collection/dumps'

topic_paths = glob.glob(os.path.join(base_dump_path, '*'))
dot = u'.'

def ws_clear(line):
    return u' '.join(line.split())

def dot_concat(line):
    line = line.split()
    buf = []

    for i in range(len(line)):
        if line[i] == dot:
            buf[-1] += dot
        else:
            buf.append(line[i])

    return u' '.join(buf)

def line_replacer(line):
    line = line.replace('.', '. ')
    line = line.replace('(', '( ')
    line = line.replace(')', ') ')

    line = line.replace(')', ') ')

    return line

def filter_sent(line):
    if len(line) < 5:
        return False
    return True

def sentence(line):
    line = ws_clear(line)
    line = dot_concat(line)

    line = line.split()
    buf, sentance = [], []

    for i in range(len(line)):
        word = line[i]
        sentance.append(word)
        if dot in word and dot == word[-1] and word not in abbreviation:
            if len(line)-1 > i:
                if line[i+1][0] in u'«1234567890QWERTYUIOPASDFGHJKLZXCVBNMЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ':
                    buf.append(sentance)
                    sentance = []
            else:
                buf.append(sentance)
                sentance = []

    return buf

def sent_printer(ws):
    for sent in ws:
        for item in sent:
            print item + ' ',
        print

def sent_printer_file(ws, f):
    for sent in ws:
        for item in sent:
            f.write(item.encode('utf-8') + ' '),
        f.write('\n\n')

for topic_path in topic_paths:
    text_paths = glob.glob(os.path.join(topic_path, '*.txt'))
    for text_path in text_paths:
        print text_path
        lines = ' '.join(open(text_path).readlines()).decode('utf-8')
        lines = line_replacer(lines)
        lines = ws_clear(lines)
        sent = sentence(lines)
        sent = filter(filter_sent, sent)
        f = open(text_path, 'w')
        sent_printer_file(sent, f)