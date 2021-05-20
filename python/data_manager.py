import statistics
import sys

import nltk
import spacy
import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


db = pymongo.MongoClient()["MIT"]["Recipes1M+"]
data = pd.read_pickle("./recipes.pkl")
#remove stop_words and apply lemmatization
def stopWord_lemma(phrase):
    sp = spacy.load('en_core_web_sm')
    ts = [',', '.', ';', '(', ')', '?', '!', '&', '%', ':', '*', '"']
    for symbol in ts:
        phrase = phrase.replace(symbol, ' ' + symbol + ' ')
    phrase = phrase.lower()
    #remove stopowords
    phrase = remove_stopwords(phrase)
    #lemmatization
    temp = sp(phrase)
    for word in temp:
        phrase = phrase.replace(str(word), word.lemma_)
    return phrase

def counter(column):
    list = []
    IDs = [r['_id'] for r in db.find()]
    with tqdm(total=len(IDs), file=sys.stdout) as pbar:
        for obj in IDs:
            pbar.update(1)
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
#data = pd.DataFrame(list(db.find()))
#data['totIngredients'] = counter('ingredients')
#data['totInstructions'] = counter('instructions')
#data.to_pickle("recipes.pkl")

def plot_statistic(column):
    fix, ax = plt.subplots(1,3 , figsize=(10,5))
    sns.distplot(data[column].values, axlabel= column, kde= False, ax=ax[0])
    sns.distplot(data[column].values, axlabel= column, kde=True, ax=ax[1])
    bplt = sns.boxplot(data=data[column].values, linewidth=1, ax=ax[2])
    bplt.set_ylabel(column)
    plt.tight_layout()
    plt.savefig('imgs/statistic' + column)
    plt.show()

#plot_statistic('totIngredients')
#plot_statistic('totInstructions')

def tfidfVec(sentence)->np.ndarray:
    #provare un tokenizzatore diverso come spacy o gensim
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    return vectorizer.fit_transform(sentence), vectorizer

#score, vector = tfidfVec(['Cream sugar and butter together till smooth'])
#tfidf_tokens = vector.get_feature_names()
#print(tfidf_tokens)
#print(score)

def search(query, corpus):
    match = cosine_similarity(query, corpus)
    print(match)
    answers, scores = [], []
    for i, s in sorted(enumerate(match[0]), key=lambda x: -x[1]):
        print(s)
        answers.append(i)
        scores.append(s)
    return answers, scores

#TEST COSINE SIMILARITY BETWEEN ONE QUERY AND ONE DOCUMENT
query = ["Yogurt Parfaits"]
document = data['instructions'].values
dd = document[0]
ingr = dd[0]['text']

doc, vectorizer = tfidfVec([ingr])
q = vectorizer.transform(query)

a,s = search(q, doc)
print(s)
print(a)