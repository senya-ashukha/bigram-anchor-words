# -*- coding: utf-8 -*-

import os

load = ('curl "http://ru.wikipedia.org/w/index.php?title=Служебная:Экспорт'
        '&pages={url}&offset=1&action=submit&format=xml&limit=1" -o "{f_name}"')

clear = 'wp2txt -i {f_name} --no-list -o {clear_dir}'

cur_dir = ''

for i, line in enumerate(open('wiki_list.txt').readlines()):
    if line.startswith(' '):
        line = line.replace('ru.wikipedia.org/wiki/', '').strip()

        xml = os.path.join('.', 'dumps', cur_dir, str(i)+'.xml')
        clear_dir = os.path.join('.', 'dumps', cur_dir, 'clear')

        dw = load.format(url=line, f_name=xml)
        cl = clear.format(f_name=xml, clear_dir=clear_dir)

        cmd = dw + ' && ' + cl

        print cmd
        os.system(cmd)
    else:
        cur_dir = line.strip().replace(' ', '_').replace(',', '')

        if not os.path.exists(os.path.join('.', 'dumps', cur_dir)):
            os.mkdir(os.path.join('.', 'dumps', cur_dir))

        if not os.path.exists(os.path.join('.', 'dumps', cur_dir, 'clear')):
            os.mkdir(os.path.join('.', 'dumps', cur_dir, 'clear'))

os.chdir('dumps')
os.system('''
    for D in `find . -type d`;
    do
        rm -r $D/*.xml;
        cp $D/clear/* $D;
        rm -r $D/clear;
    done
''')