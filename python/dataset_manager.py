import spacy
import pymongo
from bson.objectid import ObjectId
from gensim.parsing.preprocessing import remove_stopwords

sp = spacy.load('en_core_web_sm')

db = pymongo.MongoClient()["admin"]["recipes"]
IDs = [r['_id'] for r in db.find()]

for obj in IDs:
    sample = [r['Method'] for r in db.find({"_id": ObjectId(obj)})]
    for x in sample:
        x = "".join(x)
        x = x.replace(';', '').replace('**', '').replace(',', '').replace('(', '').replace(')', '').replace('.', ' ')
        #remove stopwords
        x = remove_stopwords(x)
        #lemmatization
        xz = sp(x)
        for word in xz:
            x = x.replace(str(word), word.lemma_)
            db.update_one({"_id": obj},{"$set":{"Method": x }})