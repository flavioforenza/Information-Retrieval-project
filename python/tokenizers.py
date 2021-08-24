import spacy
from keras.preprocessing.text import text_to_word_sequence
from gensim.utils import tokenize
from nltk.tokenize import word_tokenize

sp = spacy.load('en_core_web_sm')

class Tokenizer:

    def __init__(self, name = ''):
        self.name = name

    '''
    SWITCH TOKENIZERS
    '''

    def set_name(self, name):
        self.name = name

    def spacy_tokenizer(self, text=None):
        if text == None:
            return 0
        tokens = []
        doc = sp(text)
        for token in doc:
            tokens.append(token.text)
        return tokens

    def gensim_tokenizer(self, text=None):
        if text == None:
            return 0
        tokens = []
        doc = tokenize(text)
        for token in doc:
            tokens.append(token)
        return tokens

    def get_model(self):
        switcher = {
            'spacy': self.spacy_tokenizer,
            'gensim': self.gensim_tokenizer,
            'keras': text_to_word_sequence,
            'nltk': word_tokenize
        }
        return switcher.get(self.name, lambda: 'Invalid tokenizer')