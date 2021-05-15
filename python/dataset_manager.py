import spacy
import pymongo
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords
import matplotlib.pyplot as plt
import pandas as pd

#remove stop_words and apply lemmatization
def stopWord_lemma(phrase):
    phrase = remove_stopwords(phrase)
    temp = sp(phrase)
    for word in temp:
        phrase = phrase.replace(str(word), word.lemma_)
    return phrase

sp = spacy.load('en_core_web_sm')

db = pymongo.MongoClient()["admin"]["recipes"]
IDs = [r['_id'] for r in db.find()]

for obj in IDs:
    sample = [r['Method'] for r in db.find({"_id": ObjectId(obj)})]
    for x in sample:
        x = "".join(x)
        x = x.replace(';', '').replace('**', '').replace(',', '').replace('(', '').replace(')', '').replace('.', ' ')
        x = stopWord_lemma(x)
        db.update_one({"_id": obj}, {"$set": {"Method": x}})

##statistics
