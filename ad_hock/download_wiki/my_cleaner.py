# -*- coding: utf-8 -*-

import glob
import os

base_dump_path = '/Users/ars/Dropbox/Projests/Python/PycharmProjects/tmtk/ad_hock/download_wiki/data/dumps'

topic_paths = glob.glob(os.path.join(base_dump_path, '*'))

def check(line):
     return (len(line) > 0 and line[0] not in ['=', '{', '[', '|', '}', ' ', '(', ')'])

def transform(line):
    line = line.replace('{{-1|}}', ' ')
    line = line.replace('[/ref]', ' ')
    line = line.replace('[/ref]}}', ' ')
    line = line.replace('}}', ' ')
    line = line.replace('{{', ' ')
    line = line.replace('́', '')

    if '|' in line:
        line = line.split('|')[-1]

    if '.' not in line:
        line = ''

    if len(line.split()) < 10:
        line = ''

    if 'http' in line:
        buf = []
        for l in line.split():
            if 'http' not in l:
                buf.append(l)
        line = ' '.join(buf)

    buf = []
    for l in line.split():
        if l[0] not in ['-', '+']:
            buf.append(l)
    line = ' '.join(buf)

    line = line.replace('[]', '')
    line = line.replace('()', '')
    line = line.replace('ё', 'е')
    return line

for topic_path in topic_paths:
    text_paths = glob.glob(os.path.join(topic_path, '*.txt'))
    for text_path in text_paths:
        text_buf = []
        lines = open(text_path).readlines()
        for line in lines:
            if check(line):
                line = transform(line)
                if len(line):
                    text_buf.append(line)
        text_buf = '\n'.join(text_buf)
        out = open(text_path, 'w')
        out.write(text_buf)