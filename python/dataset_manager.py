import statistics

import spacy
import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns

db = pymongo.MongoClient()["MIT"]["Recipes1M+"]

#remove stop_words and apply lemmatization
def stopWord_lemma(phrase):
    sp = spacy.load('en_core_web_sm')
    ts = [',', '.', ';', '(', ')', '?', '!', '&', '%', ':', '*', '"']
    for symbol in ts:
        phrase = phrase.replace(symbol, ' ' + symbol + ' ')
    phrase = phrase.lower()
    phrase = remove_stopwords(phrase)
    temp = sp(phrase)
    for word in temp:
        phrase = phrase.replace(str(word), word.lemma_)
    return phrase

def counter(column):
    list = []
    IDs = [r['_id'] for r in db.find()]
    for obj in IDs:
        target = [r[column] for r in db.find({"_id": ObjectId(obj)})]
        for x in target:
            count = 0
            if column == "ingredients":
                count += str(x).count(",") + 1
            else:
                count += len(x)
        list.append(count)
    return list

# FOR DEBUGGING
# data = pd.DataFrame(list(db.find()))
# data['totIngredients'] = counter('ingredients')
# data['totInstructions'] = counter('instructions')
# data.to_pickle("recipes.pkl")

def plot_statistic(column):
    data = pd.read_pickle("./recipes.pkl")
    fix, ax = plt.subplots(2,3, figsize=(10,10))
    pos = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    count = 0
    for mod in column:
        sns.distplot(data[mod].values, axlabel= mod, kde= False, ax=ax[pos[count]])
        count += 1
        sns.distplot(data[mod].values, axlabel= mod, kde=True, ax=ax[pos[count]])
        count += 1
        bbx = sns.boxplot(y=data[mod].values, linewidth=1, ax=ax[pos[count]])
        count += 1
    plt.tight_layout()
    plt.savefig('imgs/displot')
    plt.show()

    #DEBUG
    #sns.boxplot(y=counter(column))

data = pd.read_pickle("./recipes.pkl")
fr = data['totIngredients'].values
print("Median:", statistics.median(fr))
print("Max:", fr.max())
print("Mean:", statistics.mean(fr))
sc = data['totInstructions'].values
print("Median:", statistics.median(sc))
print("Max:", sc.max())
print("Mean:", statistics.mean(sc))
plot_statistic(['totIngredients', 'totInstructions'])
#plot_statistic('totInstructions')


