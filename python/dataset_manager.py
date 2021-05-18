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

correction = 0

#remove stop_words and apply lemmatization
def stopWord_lemma(phrase):
    ts = [',', '.', ';', '(', ')', '?', '!', '&', '%', ':', '*', '"']
    for symbol in ts:
        phrase = phrase.replace(symbol, ' ' + symbol + ' ')
    phrase = phrase.lower()
    phrase = remove_stopwords(phrase)
    temp = sp(phrase)
    for word in temp:
        phrase = phrase.replace(str(word), word.lemma_)
    return phrase

columns = ['_id', 'Name', 'Description', 'Author', 'Ingredients', 'Method']

sp = spacy.load('en_core_web_sm')
db = pymongo.MongoClient()["MIT"]["Recipes1M+"]

#recipes_ingredients = defaultdict(dict)
values = []

if correction:
    IDs = [r['_id'] for r in db.find()]
    for obj in IDs:
        ingredients = [r['ingredients'] for r in db.find({"_id": ObjectId(obj)})]
        #list of dictionary
        for x in ingredients:
            # max_val = len(x)
            # p = tqdm(total=max_val, disable=False)
            #counting number of ingredients for every recipe
            count = 0
            for a in x:
                ingr = str(a.values())
                count += ingr.count(",") + 1
            values.append(count)
            #recipes_ingredients[ObjectId(obj)] = count
                # p.update(1)
        # for x in sample:
        #     x = "".join(x)
        #     x = stopWord_lemma(x)
        #     db.update_one({"_id": obj}, {"$set": {"Method": x}})
    data = pd.DataFrame(list(db.find()))
    data['totIngredients'] = values
    data.to_pickle("./dummy.pkl")
#print(data.isnull().sum())
data = pd.read_pickle("./dummy.pkl")
sorted = data.sort_values(by=['totIngredients'])
#meanINgredients = sum(sorted['totIngredients'])/len(sorted['totIngredients'])

# plt.figure(figsize=(16,3))
# sns.displot(data['totIngredients'], kde=False, color="#336600")
# plt.savefig('imgs/disp1.png')
# plt.show()
# plt.figure(figsize=(16,3))
# sns.displot(data['totIngredients'], kde=True, color="#336600")
# plt.show()
# plt.savefig('imgs/disp2.png')
plt.figure(figsize=(5,5))
sns.boxplot(x=data['totIngredients'])
plt.savefig('imgs/boxplot.png')
#sns.swarmplot(y=data['totIngredients'], ax=ax[2])
#plt.tight_layout()



#hist(db.find().count(), weights=data.totIngredients)


# #remove document with fields = null
# data.to_csv (r'/Users/flavioforenza/Desktop/export_dataframe.csv', index = False, header=True)
#
# data.describe().T
# #print("Description:", data.index)

