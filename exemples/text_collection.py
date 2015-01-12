# -*- coding: utf-8 -*-

from tmtk.utils import pickle

from tmtk.collection import transformer, collection

raw_data = [
    u'Зайку бросила хозяйка. Под дождем остался зайка. Со скамейки слезть не мог, весь до ниточки промок.',
    u'Нас двое в комнате: собака моя и я. На дворе воет страшная, неистовая буря.',
    u'Она словно хочет сказать мне что-то. Она немая, она без слов, она сама себя не понимает – но я ее понимаю.',
    u'Очень коротко и очень емко. Это гениально.',
    u'Мы с вами пишем блоги. Не классику. Нам нужно сказать максимум информации за короткое время.'
]

collections = collection.StringCollections(collection_name='text_collection', raw_data=raw_data)
collections.fill()

transformers = transformer.TransformerApplyer([
    transformer.PunctuationRemoverTransform(),
    transformer.LoweCaseTransform(),
    transformer.WordNormalizerTransform(core=3),
    transformer.StopWordsRemoverTransform(),
    transformer.ShortSentRemoverTransform(),
    transformer.BigramExtractorDocumentsTransform(),

    transformer.MemOptimizeTransform(),
    transformer.CreateWCMatrixTransform()

], verbose=True)

transformers.apply(collections)

pickle.dump(collections)
print pickle.load(collections.collection_name)

