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

res = []
category = np.unique(category).tolist()
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
    list_all_categories = []
    list_scores = []
    for k,v in answers:
        list_cc=[]
        increment_bar +=1
        if v>thr:
            row = (data.loc[data['id'] == k[0]])
            url = row['url'].values
            catCook = getEntity([url[0]], sum(i > thr for k,i in answers), increment_bar)
            for index, categ in catCook.items():
                print(categ)
                if categ:
                    list_cc.append(categ)
            list_all_categories.append(list_cc)
            list_scores.append(v)
    return list_all_categories, list_scores

list_all_categories,  list_scores = search_DocCategories(0.29)

'''
SEARCH CATEGORY CORRESPONDENCE - 1 METHOD
'''
y_pred = []
for x in category:
    for y in list_all_categories:
        #valuto come errate quelle senza categoria
        if not y:
            y_pred.append(1)
        for single in y:
            if x in np.unique(y).tolist():
                y_pred.append(1)
            else:
                y_pred.append(0)

#print("Documenti estratti: ", doc_Buoni)
#print("Documenti scartati: ", sum(i > thr for k, i in answers) - doc_Buoni)

precision, recall, thresholds = precision_recall_curve(y_pred, list_scores)
print("Thr:", thresholds)
fig, ax = plt.subplots()
ax.plot(recall, precision)
plt.show()
plt.close()

I = []
for i, p in enumerate(precision):
    I.append(max(precision[:i+1]))

fig, ax = plt.subplots()
ax.plot(recall, I)
plt.show()

#EVALUATION OF RANKING
#use title and ingredients
# def ranking_evaluation(query, column):
#     lst_query = []
#     relevant_doc, query = ranking(query)
#     print(query)
#     query = nltk.word_tokenize(query)
#     for k,v in relevant_doc.items():
#         lst_tokens = []
#         text = data.loc[data['_id'] == ObjectId(k), [column]]
#         values = text[column].values
#         try:
#             for lst in values:
#                 for dict in lst:
#                     for k, v in dict.items():
#                         lst_tokens.append(v + '\n')
#         except:
#             lst_tokens = values
#         lst_tokens = "".join(lst_tokens)
#         lst_tokens = clean_normalize(lst_tokens)
#         tokenize_list = nltk.word_tokenize(lst_tokens)
#         lst_result = []
#         for x in query:
#             result = 0
#             if any(x in s for s in tokenize_list):
#                 result = 1
#             lst_result.append(result)
#         lst_query.append(lst_result)
#     return lst_query, relevant_doc

#search if a column contain the term of a query








