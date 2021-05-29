import statistics
import sys
import string
import nltk
import spacy
import pymongo
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
import scrape_schema_recipe as scr
import random
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_curve
from usda.client import UsdaClient
import requests
import json


key_USDA = UsdaClient('F8TD5aG6YzDftMjk97xlAVNHnhRrnsFxSD94CRGY')

categories = ['main course', 'snack', 'soup', 'beverage', 'soup', 'stew', 'bread', 'salad', 'appetizer', 'side dish', 'dessert']
db = pymongo.MongoClient()["MIT"]["Recipes1M+"]
data = pd.read_pickle("./recipes.pkl")
IDs = [r['_id'] for r in db.find()]
sp = spacy.load('en_core_web_sm')


'''
PLOT STATISTICS OF DATASET
'''

def counter(column):
    list = []
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

'''
'''

#drop stop_words, punctuation and apply lemmatization
def clean_normalize(phrase):
    ts = [',', '.', ';', '(', ')', '?', '!', '&', '%', ':', '*', '"', '-']
    for symbol in ts:
        phrase = phrase.replace(symbol, " ")
    phrase = phrase.lower()
    #remove stopowords
    phrase = remove_stopwords(phrase)
    #remove punctuation
    exclude = set(string.punctuation)
    phrase = "".join(ch for ch in phrase if ch not in exclude)
    #lemmatization
    temp = sp(phrase)
    for word in temp:
        phrase = phrase.replace(str(word), word.lemma_)
    return phrase

#return document id and tokens of a columns
def get_value(columns):
    #print(columns[1])
    data_col = data[columns].values
    dic_id_col = {}
    with tqdm(total=len(data_col), file=sys.stdout) as pbar:
        for id, instr in data_col:
            #contains alla arrays of each field in mogno
            list_col = []
            pbar.update(1)
            try:
                for dict in instr:
                    for k,v in dict.items():
                        list_col.append(v)
            except:
                list_col.append(instr)
            new_col = clean_normalize(str(list_col))
            new_col = "".join(new_col)
            dic_id_col[id] = new_col
        file = open("id_" + columns[1] + ".pkl", "wb")
        pickle.dump(dic_id_col, file)
        file.close()
    return dic_id_col

#get_value(['id', 'title'])
#get_value(['id', 'ingredients'])
#get_value(['id', 'instructions'])

#COMPUTE THE COSINE SIMILARITY
def search(query, corpus):
    match = cosine_similarity(query, corpus)
    d_s = {}
    for i, s in sorted(enumerate(match[0]), key=lambda x: -x[1]):
        d_s[i] = s
    return d_s

def ranking(query):
    with open('id_instructions.pkl', 'rb') as f:
        id_instr = pickle.load(f)
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    doc = vectorizer.fit_transform(id_instr.values())
    #print(query)
    q = vectorizer.transform([query])
    #dict of relevant documents and scores
    doc_score = search(q, doc)
    relevant_doc = {}
    for k,v in doc_score.items():
        if v > 0:
            relevant_doc[data.loc[k]['id']] = v
    return relevant_doc, doc_score


def get_corrispondence(query, column):
    try:
        with open('id_' + column +'.pkl', 'rb') as f:
            id_column = pickle.load(f)
    except:
        pass
    ids = []
    query_tokens = nltk.word_tokenize(query)
    for k, v in id_column.items():
        doc_tokens = nltk.word_tokenize(v)
        for tk in query_tokens:
            if any(tk in s for s in doc_tokens):
                ids.append(k)
    return ids

def alterQuery():
    id_fQ = []
    with tqdm(total=len(data['title']), file=sys.stdout) as pbar:
        for title in data['title']:
            pbar.update(1)
            query = sp(title.lower())
            list_noun = []
            for token in query:
                if token.pos_ == 'NOUN':
                    list_noun.append(token.text)
            if not list_noun:
                list_noun.append(query)
            if len(list_noun)>1:
                id_fQ.append(' '.join(list_noun))
            else:
                id_fQ.append(list_noun[0])
    # file = open("fakeQuery.pkl", "wb")
    # pickle.dump(id_fQ, file)
    # file.close()
    # return id_fQ

#lst_query = alterQuery()

#now data contain fake queries
with open('dataFrame.pkl', 'rb') as f:
        data = pickle.load(f)
# adding column to dataframe
#data['Query'] = fq
# file = open("dataFrame.pkl", "wb")
# pickle.dump(data, file)
# file.close()

def getEntity(urls, pbarLen, incr):
    categories_find = {}
    with tqdm(total=pbarLen, file=sys.stdout) as pbar:
        pbar.write("Search categories")
        for i in range(len(urls)):
            pbar.update(incr)
            try:
                recipe_list = scr.scrape_url(urls[i], python_objects=True)
                poss_categ = []
                for x in categories:
                    if x in str(recipe_list[0].values()).lower():
                            poss_categ.append(x)
                categories_find[i] = poss_categ
            except:
                pass
    return categories_find

#take a random query from those available
category = []
query = ""
while not category:
    rnd = random.randint(0,len(data))
    rnd = 11
    query = data.iloc[rnd]['Query']
    query = clean_normalize(str(query))
    #extract items from web scraping
    cat = getEntity([data.iloc[rnd]['url']], 1, 1)
    for x, v in cat.items():
        if v:
            category.append(v)

queryCat = np.unique(category).tolist()
print(category)
print(query)

'''
COMPUTE TFIDFVECTORIZE AND COSINE SIMILARITY
'''
dict_score, indexDoc_score = ranking(query)
answers = [(data.loc[[i]]['id'].values, w) for i,w in sorted(enumerate(indexDoc_score.values()), key=lambda x: -x[-1])]

'''
SEARCH DOCUMENT CATEGORY
'''
def search_DocCategories(thr):
    #lista con documenti e cagtegorie(non vuote)
    print("Documenti rilevanti: ", sum(i > thr for k, i in answers))
    increment_bar = 0
    cat_not_empty = pd.DataFrame(columns=["Doc_id","Categories","Score"])
    cat_some_empty = pd.DataFrame(columns=["Doc_id", "Categories", "Score"])
    for doc_id, score in answers:
        increment_bar +=1
        if score>thr:
            row = (data.loc[data['id'] == doc_id[0]])
            url = row['url'].values
            catCook = getEntity([url[0]], sum(i > thr for k,i in answers), increment_bar)
            for index, categ in catCook.items():
                print(categ)
                if categ:
                    cat_not_empty = cat_not_empty.append({"Doc_id":doc_id,
                                                        "Categories":categ,
                                                        "Score":score},
                                                       ignore_index=True)
                cat_some_empty = cat_some_empty.append({"Doc_id":doc_id,
                                                    "Categories":categ,
                                                    "Score":score},
                                                   ignore_index=True)

    return cat_not_empty, cat_some_empty

threshold = 0.29
docCat , docCat_some_empty= search_DocCategories(threshold)

docCat_some_empty.to_pickle("./some_empty.pkl")
docCat.to_pickle("./no_empty.pkl")

docCat_some_empty = pd.read_pickle("./some_empty.pkl")
docCat = pd.read_pickle("./no_empty.pkl")
'''
SEARCH CATEGORY CORRESPONDENCE - 1 METHOD
'''
def getCatCorrispondece(qC, dC, estimate):
    y_pred = []
    for x in qC:
        #valuto come errate quelle senza categoria
        for docC in dC:
            if not docC:
                y_pred.append(estimate)
                continue
            if x in np.unique(docC).tolist():
                y_pred.append(1)
            else:
                y_pred.append(0)
    return y_pred

def plot(precision, recall, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.plot(recall, precision)
    I = []
    for i, p in enumerate(precision):
        I.append(max(precision[:i+1]))
    ax2.plot(recall, I)
    plt.show()
    plt.draw()

print(docCat_some_empty['Categories'].values)
print(docCat['Categories'].values)

'''
3 METHODS TO EVALUATE RANKING
'''
#1. DOCUMENTS WITHOUT ENTITIES = 1
estimate = 1
lol = docCat_some_empty['Categories'].values
y_pred = getCatCorrispondece(queryCat, list(docCat_some_empty['Categories'].values), estimate)
print(y_pred)
d_score = docCat_some_empty['Score'].values
precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
plot(precision, recall, 'DOCUMENTS WITHOUT ENTITIES = 1')

#2. DOCUMENTS WITHOUT ENTITIES = 0
estimate = 0
y_pred = getCatCorrispondece(queryCat, list(docCat_some_empty['Categories'].values), estimate)
print(y_pred)
d_score = docCat_some_empty['Score'].values
precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
plot(precision, recall, 'DOCUMENTS WITHOUT ENTITIES = 0')

#3. DISCARD DOCUMENTS WITHOUT ENTITIES
y_pred = getCatCorrispondece(queryCat, list(docCat['Categories'].values), estimate)
print(y_pred)
d_score = docCat['Score'].values
precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
plot(precision, recall, 'DISCARD DOCUMENTS WITHOUT ENTITIES')









