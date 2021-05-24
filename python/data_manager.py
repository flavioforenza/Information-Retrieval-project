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
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_curve


db = pymongo.MongoClient()["MIT"]["Recipes1M+"]
data = pd.read_pickle("./recipes.pkl")
IDs = [r['_id'] for r in db.find()]

#remove stop_words and apply lemmatization
def stopWord_lemma(phrase):
    sp = spacy.load('en_core_web_sm')
    ts = [',', '.', ';', '(', ')', '?', '!', '&', '%', ':', '*', '"', '-']
    for symbol in ts:
        phrase = phrase.replace(symbol, ' ' + symbol + ' ')
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

#return id and toekns of a columns
def data_processing(columns):
    data_col = data[columns].values
    dic_id_instr = {}
    with tqdm(total=len(data_col), file=sys.stdout) as pbar:
        for id, instr in data_col:
            list_instr = []
            pbar.update(1)
            for dict in instr:
                for k,v in dict.items():
                    list_instr.append(v)
            list_instr = "".join(list_instr)
            dic_id_instr[id] = stopWord_lemma(list_instr)
    return dic_id_instr
    # file = open("id_instructions.pkl", "wb")
    # pickle.dump(dic_id_instr, file)
    # file.close()

#data_processing(['_id', 'instructions'])

def search(query, corpus):
    match = cosine_similarity(query, corpus)
    d_s = {}
    for i, s in sorted(enumerate(match[0]), key=lambda x: -x[1]):
        d_s[i] = s
    return d_s

def tfidfVec(sentence: list[str])->np.ndarray:
    #provare un tokenizzatore diverso come spacy o gensim
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    return vectorizer.fit_transform(sentence), vectorizer

with open('id_instructions.pkl', 'rb') as f:
    id_instr = pickle.load(f)

#INSERIRE PARAM: QUERY
def ranking(query):
    doc, vectorizer = tfidfVec(id_instr.values())
    #TEST COSINE SIMILARITY BETWEEN ONE QUERY AND ONE DOCUMENT
    query = stopWord_lemma(query)
    #print(query)
    q = vectorizer.transform([query])
    #dict of relevant document
    doc_score = search(q, doc)
    relevant_doc = {}
    for k,v in doc_score.items():
        if v > 0.35:
            relevant_doc[data.loc[k]['_id']] = v
    return relevant_doc, query, relevant_doc

#EVALUATION OF RANKING
#use title and ingredients
def ranking_evaluation(query, column):
    lst_query = []
    relevant_doc, query, relevant_doc = ranking(query)
    print(query)
    query = nltk.word_tokenize(query)
    for k,v in relevant_doc.items():
        lst_tokens = []
        text = data.loc[data['_id'] == ObjectId(k), [column]]
        values = text[column].values
        try:
            for lst in values:
                for dict in lst:
                    for k, v in dict.items():
                        lst_tokens.append(v + '\n')
        except:
            lst_tokens = values
        lst_tokens = "".join(lst_tokens)
        lst_tokens = stopWord_lemma(lst_tokens)
        tokenize_list = nltk.word_tokenize(lst_tokens)
        lst_result = []
        for x in query:
            result = 0
            if any(x in s for s in tokenize_list):
                result = 1
            lst_result.append(result)
        lst_query.append(lst_result)
    return lst_query, relevant_doc


query = "lasagna with tomato and mozzarella"
#OR
#occ_q_i = ranking_evaluation(query, 'ingredients')
occ_q_t, relevant_doc = ranking_evaluation(query, 'title')
y_pred = []
for i in range(len(occ_q_t)):
    if occ_q_t[i].count(1)>0:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(type(y_pred))
print(type(relevant_doc.values()))
precision, recall, thresholds = precision_recall_curve(y_pred, list(relevant_doc.values()))

I = []
for i, p in enumerate(precision):
    I.append(max(precision[:i+1]))

fig, ax = plt.subplots()
ax.plot(recall, I)
plt.show()












