import sys
import pickle
import string
import sys
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
import requests
import scrape_schema_recipe as scr
import seaborn as sns
import spacy
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from keras.preprocessing.text import text_to_word_sequence
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from usda.client import UsdaClient

key_USDA = UsdaClient('F8TD5aG6YzDftMjk97xlAVNHnhRrnsFxSD94CRGY')
data = pd.read_pickle("./CustomRecipesFilter.pkl")
data.index = range(0,len(data))
categories = ['main course', 'snack', 'soup', 'beverage', 'soup', 'stew', 'bread', 'salad', 'appetizer', 'side dish', 'dessert']
db = pymongo.MongoClient()["MIT"]["Recipes1M+"]
sp = spacy.load('en_core_web_sm')
#sp = spacy.load('xx_ent_wiki_sm')

'''
SWITCH TOKENIZERS
'''
def spacy_tokenizer(text=None):
    if text == None:
        return 0
    tokens = []
    doc = sp(text)
    for token in doc:
        tokens.append(token.text)
    return tokens

def gensim_tokenizer(text=None):
    if text == None:
        return 0
    tokens = []
    doc = tokenize(text)
    for token in doc:
        tokens.append(token)
    return tokens

def tokenizer(token):
    switcher={
        'spacy': spacy_tokenizer,
        'gensim': gensim_tokenizer,
        'keras': text_to_word_sequence,
        'nltk': word_tokenize
    }
    return switcher.get(token, lambda:'Invalid tokenizer')

tokenizer_name = 'keras'
principal_tokenizer = tokenizer(tokenizer_name)

'''
PLOT STATISTICS OF DATASET
'''
#return number of instructions/ingredients
def counter(column):
    IDs = data['id'].values
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
    data_col = data[columns].values
    dic_id_col = {}
    with tqdm(total=len(data_col), file=sys.stdout) as pbar:
        for id, instr in data_col:
            #contains alla arrays of each field in mongo
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

#get_id_cleanTokens(['id', 'title'])
#get_id_cleanTokens(['id', 'ingredients'])
#get_id_cleanTokens(['id', 'instructions'])

#COMPUTE THE COSINE SIMILARITY
def ranking(query):
    with open('id_instructions.pkl', 'rb') as f:
        id_instr = pickle.load(f)
    vectorizer = TfidfVectorizer(tokenizer=principal_tokenizer)
    docs = vectorizer.fit_transform(id_instr.values())
    q = vectorizer.transform([query])
    #dict of relevant documents and scores
    doc_score = cosine_similarity(q, docs)
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
    return id_fQ

def getEntity_scrape(url):
    categories_find = {}
    for i in range(len(url)):
        try:
            recipe_list = scr.scrape_url(url[i], python_objects=True)
            poss_categ = []
            for x in categories:
                if x in str(recipe_list[0].values()).lower():
                        poss_categ.append(x)
            categories_find[i] = poss_categ
        except:
            categories_find[i] = []
    return categories_find

#take a random query from those available
def rnd_query():
    category = []
    query = ""
    print("Looking for a query...")
    while not category:
        # = random.randint(0, len(data))
        #rnd = 48566
        #rnd = 34582 #Interpolation
        rnd = 11
        #rnd = 37384
        #rnd = 16068
        #rnd = 13037
        #rnd = 39800
        #rnd = 12800 #9-->0 duplicati
        #rnd = 10726
        cat = [data.iloc[rnd]['Scrape']]
        if len(cat[0])==0:
            continue
        id_doc = data.iloc[rnd]['id']
        query = data.iloc[rnd]['Query']
        query = clean_normalize(str(query))
        category.append(cat)
    # if not os.path.exists('imgs/query' + str(rnd) + 'thr10'):
    #     os.makedirs('imgs/query' + str(rnd) + 'thr10')
    folder = ''
    #folder = 'imgs/query' + str(rnd) + 'thr10'
    return category, query, folder, id_doc, rnd

#MAIN
category, query, folder, id_doc, idx_q = rnd_query()
queryCat = np.unique(category).tolist()
print("Query:", query)
print("Query idx: ", idx_q)
print("Categories query: ", queryCat)
print("Id Doc: ", id_doc)

'''
COMPUTE TFIDFVECTORIZE AND COSINE SIMILARITY
'''
#MAIN
indexDoc_score = ranking(query)

#useful for obtaining the filtered ranking
doc_score = [(data.loc[[i]]['id'].values, w) for i, w in sorted(enumerate(indexDoc_score[0]), key=lambda x: -x[-1])]

#get the weight as threshold
threshold = [v for k,v in doc_score if k == id_doc]
print("Threshold/Score document: ", threshold)

for (id,w) in doc_score:
    if id == id_doc:
        print("Index with TFIDF: ", doc_score.index((id,w)))

'''
SEARCH DOCUMENT ENTITIES WITH "scrape_schema_recipe" API
'''
def search_DocCategories(thr):
    data = pd.read_pickle("./OriginalRecipes.pkl")
    increment_bar = 0
    cat_not_empty = pd.DataFrame(columns=["Doc_id","Categories","Score"])
    cat_some_empty = pd.DataFrame(columns=["Doc_id", "Categories", "Score"])
    with tqdm(total=sum(i > thr for k,i in doc_score), file=sys.stdout) as pbar:
        pbar.write("Search categories with Scrape-Schema-Recipe")
        for doc_id, score in doc_score:
            increment_bar +=1
            if score>=thr:
                row = (data.loc[data['id'] == doc_id[0]])
                url = row['url'].values
                catCook = getEntity_scrape([url[0]])
                pbar.update(1)
                for index, categ in catCook.items():
                    if categ:
                        cat_not_empty = cat_not_empty.append({"id":doc_id,
                                                            "Scrape":categ},
                                                           ignore_index=True)
                    cat_some_empty = cat_some_empty.append({"id":doc_id,
                                                        "Scrape":categ},
                                                       ignore_index=True)
    return cat_not_empty, cat_some_empty

#docCat , docCat_some_empty= search_DocCategories(threshold[0])

relevant = [k[0] for k, i in doc_score if i >= threshold]
docCat_some_empty = data[data['id'].isin(relevant)]
docCat = docCat_some_empty[docCat_some_empty['Scrape'].str.len()>0]

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle(title)
    ax1.plot(recall, precision)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    I = []
    for i, p in enumerate(precision):
        I.append(max(precision[:i+1]))
    ax2.plot(recall, I)
    ax2.set_xlabel('Interpolated Recall')
    ax2.set_ylabel('Precision')
    plt.figtext(.5,.90,'thr='+str(threshold)+ '    ' + 'query: ' + query,fontsize=10,ha='center')
    #plt.savefig('imgs/'+ folder +'.png')
    plt.show()

'''
3 ONLY WITH SCRAPE_SCHEMA_RECIPE API
'''

def getPred(estimate, title, lstDoc, queryCat):
    y_pred = getCatCorrispondece(queryCat, list(lstDoc['Scrape'].values), estimate)
    d_score = [w for id, w in doc_score if w >= threshold and lstDoc['id'].isin(id).any().any()]
    precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
    delta = np.diff(list(reversed(recall)))
    avgP = (delta*(list(reversed(precision))[:-1])).sum()
    #plot(precision, recall, title)
    return avgP

#1. DOCUMENTS WITHOUT ENTITIES = 1
estimate = 1
title = 'OVERESTIMATED SCRAPED ENTITIES = 1'
avgp = getPred(estimate, title, docCat_some_empty, queryCat)
print("AVG OVERESTIMATED: ", avgp)

#2. DOCUMENTS WITHOUT ENTITIES = 0
estimate = 0
title = 'UNDERESTIMATE SCRAPED ENTITIES = 0'
avgp = getPred(estimate, title, docCat_some_empty, queryCat)
print("AVG UNDERESTIMATE: ", avgp)

#3. DISCARD DOCUMENTS WITHOUT ENTITIES
title = 'DISCARD DOCUMENTS WITHOUT ENTITIES'
avgp = getPred(estimate, title, docCat, queryCat)
print("AVG DISCARD: ", avgp)


'''
SEARCH CATEGORY CORRESPONDENCE - 2nd METHOD
--- USE MIXED ENTITIES (SCRAPE_SCHEMA_RECIPE+USDA DATABASE) ---
'''
def get_entities_USDA(ingredients):
    lst_entities = []
    for ingredient in ingredients:
        text = list(ingredient.values())
        string_conc = text[0].replace(",", "%20")
        string_conc = string_conc.replace("/", ",")
        string_conc = string_conc.replace('"', ",")
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
    data = pd.read_pickle("./OriginalRecipes.pkl")
    idx_empty = []
    empty_list = docCat_some_empty[docCat_some_empty['Scrape'].str.len() == 0]
    with tqdm(total=len(empty_list), file=sys.stdout) as pbar:
        pbar.write("Search Entities in USDA Database for remaining documents ...")
        for doc_id in empty_list['id'].values:
            documents = data[data['id'] == doc_id]
            for index, row in documents.iterrows():
                idx_empty.append(index)
                ingredients = row['ingredients']
                lst_categories = get_entities_USDA(ingredients)
                pbar.update(1)
                #add entities USDA with scraped entities
                idx = docCat_some_empty[docCat_some_empty['id'] == doc_id].index.values
                docCat_some_empty.at[idx[0], 'USDA'] = lst_categories
    return docCat_some_empty

#nuovo dataframe contenete le categorie di USDA e del web scraping
#docCat_some_empty.to_pickle("./some_empty_USDA.pkl")

# get query category from USDA
def getEntitiesQuery_USDA():
    q = data.iloc[id_doc]['Query']
    #get ingredients from this index
    qidx = data[data["Query"]==q].index.values
    row = data.loc[qidx]
    ingredients = [i for i in row['ingredients']]
    lst_ingr_q_USDA = get_entities_USDA(ingredients[0])
    return lst_ingr_q_USDA

#ricerca delle categorie dei documenti rilevanti con scrape vuoto
#docCat_some_empty = getEntitiesDoc_USDA()
#categorie ingredienti query
#lst_ingr_q_USDA = getEntitiesQuery_USDA()

lst_ingr_q_USDA = data.iloc[idx_q]['USDA']

def evaluate_mixed_entities(lst_ingr_q_USDA, docCat_some_empty, queryCat):
    all_cat_query = queryCat+lst_ingr_q_USDA
    title = 'DOCUMENTS WITH MIXED ENTITIES (SCRAPE+USDA)'
    datatemp = pd.DataFrame(columns=['id','Scrape'])
    datatemp['id'] = docCat_some_empty['id']
    datatemp['Scrape'] = docCat_some_empty['Scrape'] + docCat_some_empty['USDA']
    avgp = getPred(0, title, datatemp, all_cat_query)
    return avgp

#attivare questo
avgp = evaluate_mixed_entities(lst_ingr_q_USDA, docCat_some_empty, queryCat)
print("Avg MIXED: ", avgp)
'''
SEARCH CATEGORY CORRESPONDENCE - 3rd METHOD
--- USE ONLY ENTITIES FROM USDA DATABASE ---
'''
doc_USDAEntity = {}
def only_USDA():
    with tqdm(total=sum(i >= threshold for k,i in doc_score), file=sys.stdout) as pbar:
        pbar.write("Search entities in USDA Database...")
        for doc_id, score in doc_score:
            #print("Entity search: document #", doc_id[0])
            if score >= threshold:
                row = (data.loc[data['id'] == doc_id[0]])
                ingredients = row['ingredients'].values
                entities = get_entities_USDA(ingredients[0])
                doc_USDAEntity[doc_id[0]] = entities
                pbar.update(1)

#attivare questo
#only_USDA()

def plot_only_USDA():
    #a_file = open("wdoc_USDAEntity.pkl", "rb")
    #doc_USDAEntity = pickle.load(a_file)
    #lst_ingr_q_USDA = pd.read_pickle("./lst_ingr_q_USDA.pkl")

    #remove warning numpy
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    y_pred = getCatCorrispondece(lst_ingr_q_USDA, list(docCat_some_empty['USDA'].values), 0)
    #y_pred = getCatCorrispondece(lst_ingr_q_USDA, list(doc_USDAEntity.values()), 0)
    d_score = [i for k,i in doc_score if i>=threshold]
    precision, recall, thresholds = precision_recall_curve(y_pred, d_score)
    title = 'ENTITIES FROM USDA DATABASE ONLY'
    #plot(precision, recall, title)
    delta = np.diff(list(reversed(recall)))
    avgP = (delta * (list(reversed(precision))[:-1])).sum()
    return avgP

#attivare questo
avgP = plot_only_USDA()
print("Avg only USDA: ", avgP)

'''
PCA
'''

def showPCA(query, all_relevant_documents):
    #all_relevant_queries = list(itertools.chain.from_iterable(i for i in all_relevant_queries))
    vectorizer = TfidfVectorizer(tokenizer=principal_tokenizer)
    documents = vectorizer.fit_transform(all_relevant_documents)
    qr = vectorizer.transform([query])
    pca = PCA(n_components=2)
    v_docs = pca.fit_transform(documents.toarray())
    v_query = pca.transform(qr.toarray())
    #prendi le entità di tutti i documenti ed enumerali
    plt.scatter(v_docs[:, 0], v_docs[:, 1], edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Accent', 10), label = 'Documents')
    plt.scatter(v_query[:, 0], v_query[:, 1], edgecolor='none', marker='*',
                cmap=plt.cm.get_cmap('Paired', 10), label = 'Query', s=300)
    plt.title('PCA' + '    thr='+str(threshold)+ '    ' + 'query: ' + query)
    plt.legend()
    #plt.savefig('imgs/'+folder+'PCA')
    plt.show()

with open('id_instructions.pkl', 'rb') as f:
    id_instr = pickle.load(f)
all_relevant_documents = []
for doc_id, score in doc_score:
    if score >= threshold:
        all_relevant_documents.append(id_instr[doc_id[0]])
if len(all_relevant_documents)>1:
    showPCA(query, all_relevant_documents)

''''
LANGUAGE MODEL
1. Infer a LM for each document (PAG. 224)
'''

# k-grams (s_min=2)
def skip(sequence, s=2):
    k_grams = []
    for i in range(len(sequence)):
        for j in range(i + 1, min(i + s, len(sequence))):
            k_grams.append((sequence[i], sequence[j]))
    return k_grams

def getLM_docs(step, documents = doc_score, thr = threshold):
    LMs_doc = {}
    ranking_id = []
    for doc, score in documents:
        LM = defaultdict(lambda: defaultdict(lambda: 0))
        if score>=thr:
            ranking_id.append(doc)
            if id_instr[doc[0]]:
                instructions = id_instr[doc[0]]
                #tokenizer
                tokens = principal_tokenizer(instructions)
                tokens = ["#S"] + tokens + ["#E"]
                for (a, b) in list(skip(tokens, step)):
                    LM[a][b] += 1
            LMs_doc[doc[0]] = LM
    return LMs_doc, ranking_id

#len of vocabulary from all models
def get_count_unique_word(LMs):
    vocabulary = []
    for key,v in LMs.items():
        for token in v:
            if token not in vocabulary:
                vocabulary.append(token)
    return len(vocabulary)

def Laplace_smooth(LMs, bgrams):
    l_singl = get_count_unique_word(LMs)
    bgr = list(bgrams)
    new_dict = {}
    for key,v in LMs.items():
        lst_prod = []
        for a, b in bgr:
            diff = (1 + v[a][b])/(sum(v[a].values()) + l_singl)
            lst_prod.append(diff)
        prod = np.prod(lst_prod)
        new_dict[key] = prod
    return new_dict

# Linear Interpolation Smoothing
# P(w|d)=\lambdaP(q|Md)+(1-\lambda)P(q|Mc)
def LinInterp_Smooth(LMdocs, qGrams, lamb, lamb2, LM_coll):
    l_singl = get_count_unique_word(LMdocs)
    new_dict = {}
    for doc_id, LM in LMdocs.items():
        bigramDoc = []
        bigramColl = []
        for a,b in qGrams:
            if LM[a][b] > 0: #bi-grams (numeratore)
                #check if denominator (unigram-->count(wi-1)) exist
                if sum(LM[a].values())>0:
                    bigramDoc.append((LM[a][b])/sum(LM[a].values()))
                    bigramColl.append(LM_coll[a][b]/sum(LM_coll[a].values()))
                    #bigramColl.append(sum(LM[a].values())/len(LM))
                else: #in this case unigram not exist --> scale down at (n−1)grams --> zero-gram 1/|V|
                    A = 1/l_singl
                    bigramColl.append(A)
                    #compute unigram of b
                    if sum(LM[b].values())>0:
                        len_LM = len(LM)
                        B = LM[b].values()/len_LM #unigram
                    else: #zero-gram
                        B = 1/l_singl
                    bigramDoc.append(A*B)
            else:
                if sum(LM[a].values())>0:
                    len_LM = len(LM)
                    A = sum(LM[a].values())/len_LM
                else: #in this case unigram not exist --> scale down at (n−1)grams --> zero-gram 1/|V|
                    A = 1/l_singl
                bigramColl.append(A) #p(w|Mc)
                    #compute unigram of b
                if sum(LM[b].values())>0:
                    len_LM = len(LM)
                    B = sum(LM[b].values())/len_LM
                else: #zero-gram
                    B = 1/l_singl
                bigramDoc.append(A*B) #p(wi-1,wi|Md)
        prodDocs = np.prod(bigramDoc) #P(q|Md) #bigram/unigram/zerogram
        prodColl = np.prod(bigramColl) #P(q|Mc) #unigram(wi-1)/zerogram
        # P(w|d)=\lambdaP(q|Md)+(1-\lambda)P(q|Mc)
        resultInterp = lamb*prodDocs+(1-lamb2)*prodColl
        new_dict[doc_id] = resultInterp
    return new_dict

#LM of entire collection
def getLM_coll(step, documents = doc_score, thr = threshold):
    LM_coll = defaultdict(lambda: defaultdict(lambda: 0))
    for doc, score in documents:
        if score>=thr:
            if id_instr[doc[0]]:
                instructions = id_instr[doc[0]]
                #tokenizer
                tokens = principal_tokenizer(instructions)
                tokens = ["#S"] + tokens + ["#E"]
                for (a, b) in list(skip(tokens, step)):
                    LM_coll[a][b] += 1
    return LM_coll

def LM_query(q):
    #tokenizer
    tokens = principal_tokenizer(q)
    tokens = ["#S"] + tokens + ["#E"]
    bigram = list(ngrams(tokens, 2))
    return bigram


def getIndexRelDoc(tmp):
    doc_rel = 0
    for (id,w) in tmp:
        if id == id_doc:
            doc_rel = tmp.index((id,w))
    return doc_rel

def optimals_parameters(bigram_q):
    dicLapl = {}
    dicInterp = {}
    for n in range(2 , 11, 1):
        LM_coll = getLM_coll(n)
        LM_d, remaining_doc = getLM_docs(n)
        #Laplace smoothing
        scoreMLE = Laplace_smooth(LM_d, bigram_q)
        tmpLaplace = sorted(scoreMLE.items(), key=lambda x: -x[-1])
        doc_rel_Lap = getIndexRelDoc(tmpLaplace)
        dicLapl[(n, tuple(tmpLaplace))] = doc_rel_Lap
        #Interpolation Smoothing
        lambda1 = 0
        lambda2 = 1
        results = {}
        for k in np.arange(0.1, 1.1, 0.1):
            scoreDc = LinInterp_Smooth(LM_d, bigram_q, lambda1, lambda2, LM_coll)
            tmpInterp = sorted(scoreDc.items(), key=lambda x: -x[-1])
            doc_rel_Int = getIndexRelDoc(tmpInterp)
            results[(lambda1, lambda2, tuple(tmpInterp))] = doc_rel_Int
            lambda1 = 0 + k
            lambda2 = 1 - k
            if lambda2==0:
                break
        minimum = min(results.items(), key=lambda x:x[1])
        dicInterp[n] = minimum
    ranking = []
    index = len(data)+1
    smoothing = ""
    ngram = 0
    l1 = 0
    l2 = 0
    for (k1,ran),v1 in dicLapl.items():
        for k2,v2 in dicInterp.items():
            if v1 < v2[1]:
                if index<v1:
                    break
                else:
                    index = v1
                    ngram = k1
                    smoothing = 'Laplace'
                    ranking = ran
            else:
                if index < v2[1]:
                    break
                else:
                    index = v2[1]
                    ngram = k2
                    smoothing = 'Interpolation'
                    l1 = v2[0][0]
                    l2 = v2[0][1]
                    ranking = v2[0][2]
    print("Best smoothing: ", smoothing)
    print("Index result: ", index)
    print("N-gram: ", ngram)
    if smoothing == 'Interpolation':
        print("Lambda1: ", l1)
        print("Lambda2: ", l2)
    return index, smoothing, ngram, l1, l2, ranking

bigram_q = LM_query(query)
index, smoothing, ngram, lmb1, lmb2, newRanking = optimals_parameters(bigram_q)

print("§§§§§§§§§§§§§§§§ COMPUTE PPMI VALUE §§§§§§§§§§§§§§§§")
#Co-Occurrence method
tokens = principal_tokenizer(query)
newRanking = list(newRanking)
relevant_documents = []
for i in range(0, index+1):
    relevant_documents.append(([newRanking[i][0]],newRanking[i][1]))
thr_new = newRanking[index][1]

LM_coll = getLM_coll(2, relevant_documents, thr_new)

#create term-term dataframe
# rows = terms of queries
# columns =terms of relevant documents
row_col = [word for word,count in LM_coll.items() if word not in ["#E", "#S"]]
for token in tokens:
    if token not in row_col and token not in ["#E", "#S"]:
        row_col.append(token)

term_term = pd.DataFrame(columns=row_col, index=row_col)

#fill dataframe
for token in row_col:
    for word, count in LM_coll[token].items():
        if word not in ["#E", "#S"]:
            term_term.iloc[term_term.index.get_loc(token), term_term.columns.get_loc(word)] = count


#add sum to dataframe of each columns
term_term = term_term.fillna(0)
sumCol = term_term.sum(axis=0)
term_term.loc["count(context)"] = term_term.sum(axis=0)
term_term["count(w)"] = term_term.sum(axis=1)
max_value = term_term["count(w)"].max()

print("Instructions: ", id_instr[id_doc])

#create a equal dataframe to term_temr with PPMI values
pmi_matrix = pd.DataFrame(columns= row_col, index=row_col)
for token in row_col:
    for context, count in LM_coll.items():
        if context not in ["#E", "#S"]:
            #print("Row: ", token, " Column: ", context)
            if term_term.iloc[term_term.index.get_loc(token)][context] != 0:
                p_wc = term_term.iloc[term_term.index.get_loc(token)][context]/max_value
                if term_term.iloc[term_term.index.get_loc(token)]["count(w)"] != 0:
                    p_w = term_term.iloc[term_term.index.get_loc(token)]["count(w)"]/max_value
                if term_term.iloc[term_term.index.get_loc("count(context)")][context] != 0:
                    p_c = term_term.iloc[term_term.index.get_loc("count(context)")][context]/max_value
                ppmi_value = max(np.log2(p_wc/(p_w*p_c)), 0)
                pmi_matrix.iloc[term_term.index.get_loc(token), term_term.columns.get_loc(context)] = ppmi_value
            else:
                pmi_matrix.iloc[term_term.index.get_loc(token), term_term.columns.get_loc(context)] = 0




#SVD/LSI for matrix factorization
matrix_tmp = np.matrix(pmi_matrix, dtype=float)
U, S, Vt = np.linalg.svd(matrix_tmp)

S2 = np.zeros((len(matrix_tmp), len(matrix_tmp)), float)
np.fill_diagonal(S2, S)
U_S= np.dot(U,S2)
S_Vt = np.dot(S2, Vt)

idxs = [pmi_matrix.index.get_loc(token) for token in tokens]

df_S_Vt = pd.DataFrame(np.array(S_Vt), columns= row_col, index=row_col)

vector_tokens_query = {k: [] for k in tokens}
for token in tokens:
    for column in pmi_matrix.columns:
        if column != token:
            score_similarity = cosine_similarity(np.matrix(pmi_matrix[token].tolist()),
                                                 np.matrix(pmi_matrix[column].tolist()))[0]
            vector_tokens_query[token].append((column, score_similarity[0]))







#data.loc[data['id'] == doc_id[0]]
vector_tokens_query = {k: [] for k in tokens}
for token in tokens:
    for column in pmi_matrix.columns:
        if column != token:
            score_similarity = cosine_similarity(np.matrix(pmi_matrix[token].tolist()),
                                                 np.matrix(pmi_matrix[column].tolist()))[0]
            vector_tokens_query[token].append((column, score_similarity[0]))

#order each word for each token in query
dict_sorted = {k: [] for k in tokens}
for key, value in vector_tokens_query.items():
    dict_sorted[key].append(sorted(value, key=lambda x:-x[-1]))

#creare una lista di query. Partire dalla query originale ed effettuare delle combinazioni con le parole occorrenti
all_query = {k: [] for k in tokens}
for token, list_words in dict_sorted.items():
    for (word, score) in list_words[0]:
        if score>0:
            support = []
            support = tokens.copy()
            if word not in support:
                idx = support.index(token)
                support.insert(idx+1, word)
                new_query = ' '.join(support)
                all_query[token].append(new_query)

#interpolare tutte le combinazioni di query ottenute da ogni token di codesta
#la prima lista può essere vuota!!

idx = 0
first_queries = all_query[tokens[idx]].copy()
while not first_queries:
    idx+=1
    first_queries = all_query[tokens[idx]]

for i in range (idx+1, len(all_query.keys())):
    #copia delle prime query
    temp = first_queries.copy() #lista di stringhe
    tq_w = dict_sorted[tokens[i]] #next token words
    if not tq_w[0]:
        break
    for single_query in temp:
        #tokenizzo la single_query
        tks_query = principal_tokenizer(single_query)
        #t = co-occurrence word, w: weight
        for (t,w) in tq_w[0]:
            if w>0:
                query_copy = tks_query.copy()
                idx = query_copy.index(tokens[i])
                query_copy.insert(idx+1,t)
                new_query = ' '.join(query_copy)
                first_queries.append(new_query)

for q in first_queries:
    print(q)





#k: token_query, v: list of tokens that occurs
# result = {k: [] for k in tokens}
# for token in tokens:
#     other_words = {token:[]} #può avere la lista vuota
#     for word, count in LM_coll[token].items():
#         other_words[token].append((word,count))
#         #print(token, " ", word, ": ", count)
#     #print("Len other_words: ", len(other_words[token]))
#     if len(other_words[token])>0:
#         max_value = max(other_words[token], key=lambda x: x[1])
#         remaining = [(word, count) for (word, count) in other_words[token] if count==max_value[1]]
#         #print("Len remaining: ", len(remaining))
#         result[token].append([word for (word,count) in remaining])







# #query expansion
# tmp_query = []
# for token, word_exp in result.items():
#     if len(word_exp)>0:
#         for i in word_exp:
#             if len(i)==1:
#                 new_query = token + " " + i[0]
#                 tmp_query.append(new_query)
#             elif len(i)>1:
#                 for k in i:
#                     new_query = token + " " + k
#                     tmp_query.append(new_query)
#     else:
#         tmp_query.append(token)
#
# some_queries = []
# final_query = ""
# #contiene le restanti parole da unire a una parola nella query
# lst_other = []
# for strings in tmp_query:
#     #tokenizer
#     words = principal_tokenizer(strings)
#     if words[0] not in final_query:
#         #se vi è una word in coppia
#         if len(words)>1:
#             if words[1] not in ["#E", "#", "E"]:
#                 if final_query=="":
#                     final_query += strings
#                 else:
#                     final_query += " " + strings
#         else:
#             final_query += " " + strings
#     else:
#         lst_other.append(strings)
# some_queries.append(final_query)
#
# for oth in lst_other:
#     len_some = len(some_queries)
#     while len_some!=0:
#         tmp = principal_tokenizer(some_queries[len_some-1])
#         strings = principal_tokenizer(oth)
#         #string[0] == token in query
#         if strings[0] in tmp:
#             #check boundary
#             if tmp.index(strings[0]) == len(tmp)-1 or tmp[tmp.index(strings[0])+1] not in lst_other:
#                 temp_list = []
#                 new_query = ""
#                 for i in tmp:
#                     temp_list.append(i)
#                     if i == tmp[tmp.index(strings[0])]:
#                         temp_list.append(strings[1])
#                 for i in temp_list:
#                     if new_query == "":  # remove initial space
#                         new_query += i
#                         final_query = new_query
#                     else:
#                         new_query += " " + i
#                         final_query = new_query
#                 some_queries.append(new_query)
#                 len_some -= 1
#             else:
#                 tmp[tmp.index(strings[0])+1] = strings[1]
#                 new_query = ""
#                 for i in tmp:
#                     if new_query == "":
#                         new_query += i
#                         final_query = new_query
#                     else:
#                         new_query += " " + i
#                         final_query = new_query
#                 some_queries.append(new_query)
#                 len_some -= 1
#
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print("Query generate: ", len(some_queries))
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#
# print("Instructions: ", id_instr[id_doc])
#
# def clean_query():
#     for q in some_queries:
#         tmpq = q
#         tkns = text_to_word_sequence(q)
#         for tkn in tkns:
#             if tkn in ['e', 'f', '#E', '#S' ]:
#                 tmpq.replace(tkn, '')
#                 some_queries[some_queries.index(q)] = tmpq
#             # if (tkns.count(tkn)>1):
#             #     indices = [i for i, x in enumerate(tkns) if x == tkn]
#             #     for i in indices:
#             #         if indices.index(i)>0:
#             #             del tkns[i]
#
# #clean_query()
#
# for q in some_queries:
#     #switching tokenizer is possibile to find white space in query
#     if q.find(query)==-1:
#         print("##################################")
#         print("Query: ", q)
#         bigram_q = LM_query(q)
#         index, smoothing, ngram, lmb1, lmb2, newRanking = optimals_parameters(bigram_q)








