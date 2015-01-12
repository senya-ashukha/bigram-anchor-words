import pymorphy2

def doc_normalizer(document):
    analyzer = pymorphy2.MorphAnalyzer()
    get_normal = lambda x: analyzer.parse(x)[0].normal_form
    return [map(get_normal, sent) for sent in document]