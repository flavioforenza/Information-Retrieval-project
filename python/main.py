import spacy
import nltk
import snowballstemmer
import pymongo
import gensim
import pandas
from tqdm.notebook import tqdm
#nltk.download('punkt')
#nltk.download('wordnet')
from bson.objectid import ObjectId
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from gensim.parsing.preprocessing import remove_stopwords

stemmer = snowballstemmer.stemmer('english')
sp = spacy.load('en_core_web_sm')

db = pymongo.MongoClient()["admin"]["recipes"]
IDs = [r['_id'] for r in db.find()]

for obj in IDs:
    sample = [r['Method'] for r in db.find({"_id": ObjectId(obj)})]
    for x in sample:
        x = "".join(x)
        x = x.replace(';', '').replace('**', '').replace(',', '').replace('(', '').replace(')', '').replace('.', ' ')
        #remove stopwords
        x = remove_stopwords(x)
        #lemmatization
        xz = sp(x)
        for word in xz:
            x = x.replace(str(word), word.lemma_)
            db.update_one({"_id": obj},{"$set":{"Method": x }})

methods = [r['Method'] for r in db.find()]
I = defaultdict(lambda: defaultdict(lambda:0))
for line in methods:
    for sentence in sent_tokenize(line):
        tokens = ['#S'] + word_tokenize(sentence) + ['#F']
        for (a, b) in nltk.ngrams(tokens, 2):
            I[a][b] += 1

query = "cut each and good"
query = remove_stopwords(query)
queryX = sp(query)
for word in queryX:
    query = query.replace(str(word), word.lemma_)

bgrams = nltk.ngrams(['#S'] + word_tokenize(query) + ['#F'], 2)
print([I[a][b] / sum(I[a].values()) for a, b in bgrams])
