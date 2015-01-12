import urllib2
from BeautifulSoup import BeautifulSoup

page = urllib2.urlopen('https://ru.wikipedia.org/wiki/%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F:%D0%98%D0%B7%D0%B1%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5_%D1%81%D1%82%D0%B0%D1%82%D1%8C%D0%B8')
soup = BeautifulSoup(page)

tds = [
    soup.find('td', attrs={'style': 'vertical-align:top; padding-right: 1em; width:50%'}),
    soup.find('td', attrs={'style': 'vertical-align:top;'})
]

bsns = "<class 'BeautifulSoup.NavigableString'>"

def norm(num, s):
    if s[-1] == ':': s=s[:-1]
    return '\t'*num + s

for td in tds:
    for first_lavel_tag in td:
        if str(type(first_lavel_tag)) == bsns: continue
        if 'style' in dict(first_lavel_tag.attrs) and \
            dict(first_lavel_tag.attrs)['style'] == 'border-bottom: solid 2px #479429;':
            pass#print first_lavel_tag.text

        if first_lavel_tag.name == 'p':
            for second_lavel_tag in first_lavel_tag:
                if str(type(second_lavel_tag)) == bsns: continue
                if second_lavel_tag.name == 'b':
                    print norm(0, second_lavel_tag.text)
                if second_lavel_tag.name == 'a':
                    print 1*'\t' + 'ru.wikipedia.org'+ dict(second_lavel_tag.attrs)['href']

        if first_lavel_tag.name =='ul':
            a = first_lavel_tag
            for first_lavel_tag in a:
                if str(type(first_lavel_tag)) == bsns: continue
                for second_lavel_tag in first_lavel_tag:
                    if str(type(second_lavel_tag)) == bsns: continue
                    if second_lavel_tag.name == 'b':
                        pass#print norm(3, second_lavel_tag.text)
                    if second_lavel_tag.name == 'a':
                        print 1*'\t' + 'ru.wikipedia.org'+ dict(second_lavel_tag.attrs)['href']
