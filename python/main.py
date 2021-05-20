import spacy
import nltk
import pymongo
import gensim
import pandas
from bson.objectid import ObjectId
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from gensim.parsing.preprocessing import remove_stopwords
from data_manager import stopWord_lemma
from tqdm.notebook import tqdm

sp = spacy.load('en_core_web_sm')

db = pymongo.MongoClient()["Christmas"]["Recipe"]

methods = [r['Method'] for r in db.find()]
I = defaultdict(lambda: defaultdict(lambda:0))
for line in methods:
    for sentence in sent_tokenize(line):
        tokens = ['#S'] + word_tokenize(sentence) + ['#F']
        for (a, b) in nltk.ngrams(tokens, 2):
            I[a][b] += 1

query = "Cream sugar and butter together till smooth"
query = stopWord_lemma(query)
print("Query inserted:", query)

bgrams = nltk.ngrams(['#S'] + word_tokenize(query) + ['#F'], 2)

#for a,b in bgrams:
#    print(I[a][b] / sum(I[a].values()), ", " , a, ', ',b )
print([I[a][b] / sum(I[a].values()) for a, b in bgrams])
