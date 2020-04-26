import pickle
from nltk.corpus import stopwords
import spacy

from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer

stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()

stopwords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_lg")


class TextPreprocessing:

    def __init__(self, input_text):
        self.input_text = input_text

    def pos_tagging(self, text):
        doc = nlp(text)
        l = ['DATE', 'GPE']
        for ent in doc.ents:
            if ent.label_ in l:
                text = text.replace(ent.text, ent.label_)
        return text

    def tokenizer_and_pad(self):
        from keras.preprocessing.sequence import pad_sequences
        # loading
        with open('/Users/subir/Codes/Miscellaneous/AnalytixLab/Mixed - Digital - Text Classification-Python/tokenizer.pickle',
                'rb') as handle:
            tokenizer = pickle.load(handle)
        pos_tagged_text = self.pos_tagging(self.input_text)
        stemmed_text = " ".join([stemmer.stem(i) for i in pos_tagged_text.split()])
        lmtz_text = " ".join([lmtzr.lemmatize(i) for i in stemmed_text.split()])
        ip_word_seq = tokenizer.texts_to_sequences([lmtz_text])
        ip_word_seq = pad_sequences(ip_word_seq, maxlen=150)
        return ip_word_seq
