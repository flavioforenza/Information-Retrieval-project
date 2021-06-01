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
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_curve
from usda.client import UsdaClient
import requests
import json
import itertools



key_USDA = UsdaClient('F8TD5aG6YzDftMjk97xlAVNHnhRrnsFxSD94CRGY')


categories = ['main course', 'snack', 'soup', 'beverage', 'soup', 'stew', 'bread', 'salad', 'appetizer', 'side dish', 'dessert']
db = pymongo.MongoClient()["MIT"]["Recipes1M+"]
data = pd.read_pickle("./recipes.pkl")
IDs = [r['_id'] for r in db.find()]
sp = spacy.load('en_core_web_sm')
#sp = spacy.load('xx_ent_wiki_sm')


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
    # plt.savefig('imgs/statistic' + column)
    # plt.show()

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
def get_id_cleanTokens(columns):
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

#get_id_tokens(['id', 'title'])
#get_id_tokens(['id', 'ingredients'])
#get_id_tokens(['id', 'instructions'])

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
    return doc_score

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
    return id_fQ

#lst_query = alterQuery()

# adding column to dataframe
#data['Query'] = fq
# file = open("dataFrame.pkl", "wb")
# pickle.dump(data, file)
# file.close()

#now data contain fake queries
with open('dataFrame.pkl', 'rb') as f:
        data = pickle.load(f)

def getEntity_scrape(url, pbarLen, incr):
    categories_find = {}
    with tqdm(total=pbarLen, file=sys.stdout) as pbar:
        pbar.write("Search categories")
        for i in range(len(url)):
            pbar.update(incr)
            try:
                recipe_list = scr.scrape_url(url[i], python_objects=True)
                poss_categ = []
                for x in categories:
                    if x in str(recipe_list[0].values()).lower():
                            poss_categ.append(x)
                categories_find[i] = poss_categ
            except:
                pass
    return categories_find

#take a random query from those available

def rnd_query(rnd):
    category = []
    query = ""
    while not category:
        rnd = 11
        query = data.iloc[rnd]['Query']
        query = clean_normalize(str(query))
        #extract items from web scraping
        cat = getEntity_scrape([data.iloc[rnd]['url']], 1, 1)
        for x, v in cat.items():
            if v:
                category.append(v)
    return category, query

#MAIN
category, query = rnd_query(random.randint(0,len(data)))
queryCat = np.unique(category).tolist()
print(queryCat)
print(query)

'''
COMPUTE TFIDFVECTORIZE AND COSINE SIMILARITY
'''
#MAIN
indexDoc_score = ranking(query)
#useful for obtaining the filtered ranking
doc_score = [(data.loc[[i]]['id'].values, w) for i,w in sorted(enumerate(indexDoc_score.values()), key=lambda x: -x[-1])]

'''
SEARCH DOCUMENT ENTITIES WITH "scrape_schema_recipe" API
'''
def search_DocCategories(thr):
    #lista con documenti e cagtegorie(non vuote)
    #print("Documenti rilevanti: ", sum(i > thr for k, i in doc_score))
    increment_bar = 0
    cat_not_empty = pd.DataFrame(columns=["Doc_id","Categories","Score"])
    cat_some_empty = pd.DataFrame(columns=["Doc_id", "Categories", "Score"])
    for doc_id, score in doc_score:
        increment_bar +=1
        if score>thr:
            row = (data.loc[data['id'] == doc_id[0]])
            url = row['url'].values
            catCook = getEntity_scrape([url[0]], sum(i > thr for k,i in doc_score), increment_bar)
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
#docCat , docCat_some_empty= search_DocCategories(threshold)

#docCat_some_empty.to_pickle("./some_empty.pkl")
#docCat.to_pickle("./no_empty.pkl")

docCat_some_empty = pd.read_pickle("./some_empty.pkl")
docCat = pd.read_pickle("./no_empty.pkl")
'''
SEARCH CATEGORY CORRESPONDENCE - 1st METHOD
--- USE ENTITIES FROM SCRAPE_SCHEMA_RECIPE API ---
'''

def getCatCorrispondece(qC, dC, estimate):
    y_pred = []
    for docC in dC:
        if not docC:
            y_pred.append(estimate)
            continue
        try:
            corrispondence = [value for value in np.unique(qC).tolist() if value in np.unique(docC).tolist()]
        except:
            corrispondence.append([])
        #se vi è almeno una corrispondenza
        if len(corrispondence) > 0:
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

'''
3 METHODS TO EVALUATE RANKING (ONLY WITH SCRAPE_SCHEMA_RECIPE API)
'''
#1. DOCUMENTS WITHOUT ENTITIES = 1
# estimate = 1
# y_pred = getCatCorrispondece(queryCat, list(docCat_some_empty['Categories'].values), estimate)
# d_score = docCat_some_empty['Score'].values
# precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
# plot(precision, recall, 'DOCUMENTS WITHOUT ENTITIES = 1')

#2. DOCUMENTS WITHOUT ENTITIES = 0
# estimate = 0
# y_pred = getCatCorrispondece(queryCat, list(docCat_some_empty['Categories'].values), estimate)
# d_score = docCat_some_empty['Score'].values
# precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
# plot(precision, recall, 'DOCUMENTS WITHOUT ENTITIES = 0')

# #3. DISCARD DOCUMENTS WITHOUT ENTITIES
# y_pred = getCatCorrispondece(queryCat, list(docCat['Categories'].values), estimate)
# print(y_pred)
# d_score = docCat['Score'].values
# precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
# plot(precision, recall, 'DISCARD DOCUMENTS WITHOUT ENTITIES')

'''
SEARCH CATEGORY CORRESPONDENCE - 2nd METHOD
--- USE MIXED ENTITIES (SCRAPE_SCHEMA_RECIPE+USDA DATABASE) ---
'''
def get_entities_USDA(ingredients):
    lst_entities = []
    for ingredient in ingredients:
        text = list(ingredient.values())
        string_conc = text[0].replace(",", "%20")
        string_conc = string_conc.replace(" ", "")
        urlUSDA = "https://api.nal.usda.gov/fdc/v1/foods/search?api_key=" + key_USDA.key + "&query=" + string_conc
        try:
            res = requests.get(urlUSDA)
            js = res.json()
            cat_USDA = js['foods'][0]['foodCategory']
        except:
            continue
        lst_entities.append(cat_USDA)
    return lst_entities

def getEntitiesDoc_USDA():
    idx_empty = []
    empty_list = docCat_some_empty[docCat_some_empty['Categories'].str.len() == 0]
    for doc_id in empty_list['Doc_id'].values:
        documents = data[data['id'] == doc_id[0]]
        for index, row in documents.iterrows():
            idx_empty.append(index)
            ingredients = row['ingredients']
            lst_categories = get_entities_USDA(ingredients)
            #add entities USDA with scraped entities
            idx = docCat_some_empty[docCat_some_empty['Doc_id'] == doc_id[0]].index.values
            docCat_some_empty.at[idx[0], 'Categories'] = lst_categories
    return docCat_some_empty

#nuovo dataframe contenete le categorie di USDA e del web scraping
#docCat_some_empty.to_pickle("./some_empty_USDA.pkl")

# get query category from USDA
def getEntitiesQuery_USDA():
    q = data.iloc[11]['Query']
    #get ingredients from this index
    qidx = data[data["Query"]==q].index.values
    row = data.loc[qidx]
    ingredients = [i for i in row['ingredients']]
    lst_ingr_q_USDA = get_entities_USDA(ingredients[0])
    return lst_ingr_q_USDA

#save lst_ingr_q_USDA
# open_file = open("lst_ingr_q_USDA.pkl", "wb")
# pickle.dump(lst_ingr_q_USDA, open_file)
# open_file.close()

#docCat_some_empty = getEntitiesDoc_USDA()
#lst_ingr_q_USDA = getEntitiesQuery_USDA()


def evaluate_mixed_entities(lst_ingr_q_USDA, docCat_some_empty, queryCat):
    lst_ingr_q_USDA = pd.read_pickle("./lst_ingr_q_USDA.pkl")
    queryCat = pd.read_pickle("./queryCat.pkl")
    docCat_some_empty = pd.read_pickle("./some_empty_USDA.pkl")

    all_cat_query = queryCat+lst_ingr_q_USDA
    estimate = 0
    y_pred = getCatCorrispondece(all_cat_query, list(docCat_some_empty['Categories'].values), estimate)
    d_score = docCat_some_empty['Score'].values
    precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
    plot(precision, recall, 'METRICS WITH MIXED ENTITIES (SCRAPE+USDA)')

'''
SEARCH CATEGORY CORRESPONDENCE - 3rd METHOD
--- USE ONLY ENTITIES FROM USDA DATABASE ---
'''
def only_USDA():
    doc_USDAEntity = {}
    with tqdm(total=sum(i > threshold for k,i in doc_score), file=sys.stdout) as pbar:
        pbar.write("Search categories in USDA DB")
        for doc_id, score in doc_score:
            print("Entity search: document #", doc_id[0])
            if score > threshold:
                row = (data.loc[data['id'] == doc_id[0]])
                ingredients = row['ingredients'].values
                entities = get_entities_USDA(ingredients[0])
                doc_USDAEntity[doc_id[0]] = entities
                pbar.update(1)
    # a_file = open("doc_USDAEntity.pkl", "wb")
    # pickle.dump(doc_USDAEntity, a_file)
    # a_file.close()

def plot_only_USDA():
    a_file = open("doc_USDAEntity.pkl", "rb")
    doc_USDAEntity = pickle.load(a_file)
    lst_ingr_q_USDA = pd.read_pickle("./lst_ingr_q_USDA.pkl")
    #remove warning numpy
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    y_pred = getCatCorrispondece(lst_ingr_q_USDA, list(doc_USDAEntity.values()), 0)
    d_score = [i for k,i in doc_score if i>threshold]
    precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
    plot(precision, recall, 'ONLY WITH USDA ENTITIES')

#plot_only_USDA()

'''
PCA
'''

def PCA(query, doc_score):
    with open('id_instructions.pkl', 'rb') as f:
        id_instr = pickle.load(f)

    all_relevant_documents = []
    for doc_id, score in doc_score:
        if score > threshold:
            all_relevant_documents.append(id_instr[doc_id[0]])

    #all_relevant_queries = list(itertools.chain.from_iterable(i for i in all_relevant_queries))
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    documents = vectorizer.fit_transform(all_relevant_documents)
    query = vectorizer.transform([query])

    pca = PCA(n_components=2)
    v_docs = pca.fit_transform(documents.toarray())
    v_query = pca.transform(query.toarray())

    #prendi le entità di tutti i documenti ed enumerali
    plt.scatter(v_docs[:, 0], v_docs[:, 1], edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Accent', 10))
    plt.scatter(v_query[:, 0], v_query[:, 1], edgecolor='none', marker='*',
                cmap=plt.cm.get_cmap('Paired', 10))
    plt.show()

''''
LANGUAGE MODEL
1. Infer a LM for each document (PAG. 224)
'''










# all_relevant_queries = []
# for doc_id, score in doc_score:
#     if score > threshold:
#         pp = data[data['id']==doc_id[0]]['Query'].values
#         all_relevant_queries.append(str(pp[0]))