import os
import zipfile

import cPickle as pickle
from StringIO import StringIO

pickle_data_path ='/Users/ars/Dropbox/Projests/Python/PycharmProjects/tmtk/tmtk/data/pickle'

def dump(collections):
    pickle_collection = StringIO()
    pickler_collection = pickle.Pickler(pickle_collection)
    pickler_collection.dump(collections)
    zip_file = zipfile.ZipFile(
        os.path.join(pickle_data_path, collections.collection_name + '.pickle.zip'), 'w',
        compression=zipfile.ZIP_DEFLATED
    )
    zip_file.writestr('collection.pickle', pickle_collection.getvalue())

def load(name):
    with zipfile.ZipFile(open(os.path.join(pickle_data_path, name + '.pickle.zip')), mode='r') as zip_file:
        return pickle.load(StringIO(zip_file.read('collection.pickle')))