import spacy
import nltk
import pymongo
import gensim
import pandas as pd
from bson.objectid import ObjectId
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from gensim.parsing.preprocessing import remove_stopwords
import data_manager
from tqdm.notebook import tqdm
from usda.client import UsdaClient
import requests


sp = spacy.load('en_core_web_sm')
data = pd.read_pickle("./dataFrame.pkl")


'''
PLOT STATISTICS DATASET
'''
if __name__ == "__main__":
    data_manager.plot_statistic('totIngredients')
    data_manager.plot_statistic('totInstructions')

'''
GET A RANDOM QUERY
'''
categories, query = rnd_query(random.randint(0,len(data)))
queryCat = np.unique(categories).tolist()
print("Query: ", query)
print("Categories: ", categories)

'''
COMPUTE TFIDFVECTORIZE AND COSINE SIMILARITY 
TO GET RANKING OF RELEVANT DOCUMENTS
'''
indexDoc_score = ranking(query)
#LIST OF DOCUMENTS-SCORES
doc_score = [(data.loc[[i]]['id'].values, w) for i,w in sorted(enumerate(indexDoc_score.values()), key=lambda x: -x[-1])]

'''
ENTITIES 
'''
threshold = 0.29
#docCat , docCat_some_empty= search_DocCategories(threshold)
docCat = pd.read_pickle("./no_empty.pkl")
docCat_some_empty = pd.read_pickle("./some_empty.pkl")

'''
3 WAYS TO CONSIDER PREDICTIONS (OBTAINED ONLY WITH THE SCRAPE_SCHEMA_RECIPE API)
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
#entity search for documents that don't have it
docCat_some_empty = getEntitiesDoc_USDA()
lst_ingr_q_USDA = getEntitiesQuery_USDA()
#plot metrics results
evaluate_mixed_entities(docCat_some_empty,lst_ingr_q_USDA, queryCat)

'''
SEARCH CATEGORY CORRESPONDENCE - 3rd METHOD
--- USE ONLY ENTITIES FROM USDA DATABASE ---
'''
#sistemare parametri
only_USDA()



















# db = pymongo.MongoClient()["Christmas"]["Recipe"]
#
# methods = [r['Method'] for r in db.find()]
# I = defaultdict(lambda: defaultdict(lambda:0))
# for line in methods:
#     for sentence in sent_tokenize(line):
#         tokens = ['#S'] + word_tokenize(sentence) + ['#F']
#         for (a, b) in nltk.ngrams(tokens, 2):
#             I[a][b] += 1
#
# query = "Cream sugar and butter together till smooth"
# query = stopWord_lemma(query)
# print("Query inserted:", query)
#
# bgrams = nltk.ngrams(['#S'] + word_tokenize(query) + ['#F'], 2)
#
# #for a,b in bgrams:
# #    print(I[a][b] / sum(I[a].values()), ", " , a, ', ',b )
# print([I[a][b] / sum(I[a].values()) for a, b in bgrams])


