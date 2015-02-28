# -*- coding: utf-8 -*-

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

stopwords_ru = [
    u'а', u'без', u'более', u'больше', u'будет', u'будто', u'бы', u'был', u'была', u'были', u'было', u'быть', u'в',
    u'вам', u'вас', u'вдруг', u'ведь', u'во', u'вот', u'впрочем', u'все', u'всегда', u'всего', u'всех', u'всю', u'вы',
    u'г', u'где', u'говорил', u'да', u'даже', u'два', u'для', u'до', u'другой', u'его', u'ее', u'ей', u'ему', u'если',
    u'есть', u'еще', u'ж', u'же', u'жизнь', u'за', u'зачем', u'здесь', u'и', u'из', u'из-за', u'или', u'им', u'иногда',
    u'их', u'к', u'кажется', u'как', u'какая', u'какой', u'когда', u'конечно', u'которого', u'которые', u'кто', u'куда',
    u'ли', u'лучше', u'между', u'меня', u'мне', u'много', u'может', u'можно', u'мой', u'моя', u'мы', u'на', u'над',
    u'надо', u'наконец', u'нас', u'не', u'него', u'нее', u'ней', u'нельзя', u'нет', u'ни', u'нибудь', u'никогда',
    u'ним', u'них', u'ничего', u'но', u'ну', u'о', u'об', u'один', u'он', u'она', u'они', u'опять', u'от', u'перед',
    u'по', u'под', u'после', u'потом', u'потому', u'почти', u'при', u'про', u'раз', u'разве', u'с', u'сам', u'свое',
    u'свою', u'себе', u'себя', u'сегодня', u'сейчас', u'сказал', u'сказала', u'сказать', u'со', u'совсем', u'так',
    u'такой', u'там', u'тебя', u'тем', u'теперь', u'то', u'тогда', u'того', u'тоже', u'только', u'том', u'тот', u'три',
    u'тут', u'ты', u'у', u'уж', u'уже', u'хорошо', u'хоть', u'чего', u'человек', u'чем', u'через', u'что', u'чтоб',
    u'чтобы', u'чуть', u'эти', u'этого', u'этой', u'этом', u'этот', u'эту', u'это', u'я', u'весь', u'свой', u'это',
    u'весь', u'который', u'еще', u'наш', u'каждый', u'всякий', u'другой', u'мочь']

def doc_normalizer(document):
    analyzer = pymorphy2.MorphAnalyzer()
    get_normal = lambda x: analyzer.parse(x)[0].normal_form
    return [map(get_normal, sent) for sent in document]

def is_stop_words(w):
    if w in stopwords_ru:
        return True
    if morph.parse(w)[0].tag.POS in ['PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO']:
        return True
    return False

def doc_stop_word_remove(document):
    return [filter(lambda word: not is_stop_words(word), sent) for sent in document]

