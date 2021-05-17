import spacy
import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords

correction = 0

#remove stop_words and apply lemmatization
def stopWord_lemma(phrase):
    phrase = phrase.lower()
    phrase = remove_stopwords(phrase)
    temp = sp(phrase)
    for word in temp:
        phrase = phrase.replace(str(word), word.lemma_)
    return phrase

columns = ['_id', 'Name', 'Description', 'Author', 'Ingredients', 'Method']

sp = spacy.load('en_core_web_sm')

db = pymongo.MongoClient()["admin"]["recipes"]

if correction:
    IDs = [r['_id'] for r in db.find()]
    for obj in IDs:
        sample = [r['Method'] for r in db.find({"_id": ObjectId(obj)})]
        for x in sample:
            x = "".join(x)
            x = x.replace(';', '').replace('**', '').replace(',', '').replace('(', '').replace(')', '').replace('.', ' ')
            x = stopWord_lemma(x)
            db.update_one({"_id": obj}, {"$set": {"Method": x}})

data = pd.DataFrame(list(db.find({}, {'url':0})))
#remove document with fields = null
data.to_csv (r'/Users/flavioforenza/Desktop/export_dataframe.csv', index = False, header=True)

data.describe().T
#print("Description:", data.index)

