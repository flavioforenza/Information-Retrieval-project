import spacy
import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
from matplotlib.pyplot import hist
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
    sns_dis = sns.displot(data = data[column], kde=True)
    sns_dis.savefig('imgs/displot ' + column)
    plt.close()
    sns_boxplot = sns.boxplot(y=data[column], linewidth=1, orient='v')
    plt.savefig('imgs/boxplot ' + column)
    plt.close()
    #DEBUG
    #sns.boxplot(y=counter(column))

plot_statistic('totIngredients')


