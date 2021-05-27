import spacy
import nltk
import pymongo
import gensim
import pandas
from bson.objectid import ObjectId
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
from gensim.parsing.preprocessing import remove_stopwords
from data_manager import *
from tqdm.notebook import tqdm

sp = spacy.load('en_core_web_sm')

# query = "lasagna with tomato and mozzarella"
# '''
# SEARCH QUERY IN INGREDIENTS TO GET DOCUMENT ID
# '''
# query = clean_normalize(query)
# #print(query)
# list_ids_ingr = get_corrispondence(query, 'ingredients')
# list_ids_title = get_corrispondence(query, 'title')
# print("ingr:",list_ids_ingr)
# print("title:",list_ids_title)
# print("common:", list(set(list_ids_title) & set(list_ids_ingr)))

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

query = "lasagna with tomato and mozzarella"
'''
SEARCH QUERY IN INGREDIENTS TO GET DOCUMENT ID
'''
query = clean_normalize(query)
list_ids_ingr = get_corrispondence(query, 'ingredients')
list_ids_title = get_corrispondence(query, 'title')
corrisp_Intersection = list(set(list_ids_title) & set(list_ids_ingr))
corrisp_Union = list_ids_title+list_ids_ingr
#print("common:", list(set(list_ids_title) & set(list_ids_ingr)))

'''
COMPUTE TFIDFVECTORIZE AND COSINE SIMILARITY
'''
dict_score, indexDoc_score = ranking(query) #dict_score contiene i documenti con peso > 0
#print(indexDoc_score.values())
#rimuovere i documenti che non sono rilevanti (con peso = 0), successivamente vedere se questi hanno gli id corrispondenti
answers = [(data.loc[[i]]['id'].values, w) for i,w in sorted(enumerate(indexDoc_score.values()), key=lambda x: -x[-1])]
#print((data.loc[[i]]['id'].values, w) for i,w in sorted(enumerate(dict_score.values()), key=lambda x: -x[-1]))
#print(dict_score.values())

y_pred, scores = [], []
for e, score in answers:
    if e in corrisp_Union:
        y_pred.append(1)
    else:
        y_pred.append(0)
    scores.append(score)
print(y_pred)
#print("Conteggio 1:", y_pred.count(1))
#print("Conteggio 0:", y_pred.count(0))

precision, recall, thresholds = precision_recall_curve(y_pred, scores)
print(precision)

fig, ax = plt.subplots()
ax.plot(recall, precision)
plt.show()

I = []
for i, p in enumerate(precision):
    I.append(max(precision[:i+1]))

#print(I)
#
# fig, ax = plt.subplots()
# ax.plot(recall, I)
# plt.show()


