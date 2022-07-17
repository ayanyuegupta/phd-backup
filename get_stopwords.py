import spacy
import pickle

sp = spacy.load('en_core_web_sm')
stopwords = sp.Defaults.stop_words
with open('stopwords.pickle', 'wb') as f:
    pickle.dump(stopwords, f)


