from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from tmtk.utils import pickle

n_topics, n_top_words = 40, 10

collection = pickle.load('wiki_ru_featured_articles')
df = list(collection.yield_str_documents())

vectorizer = CountVectorizer(max_df=0.95, min_df=2)
tfidf = vectorizer.fit_transform(df)

nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)

feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print "Topic #%d: " % topic_idx + " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
