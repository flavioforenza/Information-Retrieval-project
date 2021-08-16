import data_manager as dm
import numpy as np
from tokenizers import Tokenizer
import pandas as pd
import pickle

data = pd.read_pickle("./CustomRecipesFilter.pkl")

dm.plot_statistic('totInstructions')
dm.plot_statistic('totIngredients')

threshold = 0
#1. Select a random query
while threshold<0.10:
    query_obj = dm.rnd_query()
    queryCat = np.unique(query_obj.categories).tolist()
    print("Query:", query_obj.query)
    print("Query idx: ", query_obj.index)
    print("Categories query: ", queryCat)
    print("Id Doc: ", query_obj.id_doc)

    #2. Choice of Tokenizer
    tokenizer = Tokenizer()
    tokenizer.set_name('keras')

    #3. Compute Tf-Idf-Vectorize and Cosine similarity
    dm.ranking(query_obj, tokenizer)
    threshold = query_obj.threshold
    print("Threshold/Score document: ", query_obj.threshold)

#4. Show index of document target
for (id, w) in query_obj.doc_score:
    if id == query_obj.id_doc:
        print("Index with TFIDF: ", query_obj.doc_score.index((id, w)))

############## RANKING ANALYSIS ##############

#5. Lists of documents with/without entities
docCat_some_empty = data[data['id'].isin(query_obj.relevant_ranking_tfidf())]
docCat = docCat_some_empty[docCat_some_empty['Scrape'].str.len() > 0]

#5.1 OVERESTIMATE DOCUMENTS WITHOUT ENTITIES = 1
estimate = 1
title = 'OVERESTIMATED SCRAPED ENTITIES = 1'
avgp = dm.getPred(estimate, title, docCat_some_empty, query_obj.categories, query_obj)
print("AVG OVERESTIMATED: ", avgp)

#5.2 UNDERESTIMATE DOCUMENT WITHOUT ENTITIES = 0
estimate = 0
title = 'UNDERESTIMATE SCRAPED ENTITIES = 0'
avgp = dm.getPred(estimate, title, docCat_some_empty, query_obj.categories, query_obj)
print("AVG UNDERESTIMATE: ", avgp)

#5.3 DISCARD DOCUMENTS WITHOUT ENTITIES
title = 'DISCARD DOCUMENTS WITHOUT ENTITIES'
avgp = dm.getPred(estimate, title, docCat, query_obj.categories, query_obj)
print("AVG DISCARD: ", avgp)

#5.4 EVALUATION WITH MIXED ENTITIES (SCRAPE + USDA)
lst_ingr_q_USDA = data.iloc[query_obj.index]['USDA']
avgp = dm.evaluate_mixed_entities(lst_ingr_q_USDA, docCat_some_empty, query_obj)
print("Avg MIXED: ", avgp)

#5.5 EVALUTATION ONLY WITH USDA ENTITIES
avgP = dm.plot_only_USDA(query_obj, lst_ingr_q_USDA, docCat_some_empty)
print("Avg only USDA: ", avgP)

#6. SHOW PRINCIPAL COMPONENT ANALYSIS (PCA)
with open('id_instructions.pkl', 'rb') as f:
    id_instr = pickle.load(f)
all_relevant_documents = []
for doc_id, score in query_obj.doc_score:
    if score >= query_obj.threshold:
        all_relevant_documents.append(id_instr[doc_id[0]])
if len(all_relevant_documents) > 1:
    dm.showPCA(query_obj, all_relevant_documents, tokenizer)

############## QUERY EXPANSION ##############

#7. EVERY RECIPE INSTRUCTION IS CONSIDERED AS A DOCUMENT
print("Instructions: ", id_instr[query_obj.id_doc])

#8. CONSTRUCTION OF THE TERM-TERM MATRIX
tokens, row_col, LM_coll, term_term, max_value = dm.term_term_matrix(query_obj, tokenizer)

#9. COMPUTATION OF POSITIVE POINTWISE-MUTUAL-INFORMATION (PPMI) ON TERM-TERM MATRIX
# BASED ON CO-OCCURRENCE METHOD
Pmi_matrix = dm.pmi_matrix(row_col, LM_coll, term_term, max_value)

#10. COMPUTATION OF SINGULAR VALUE DECOMPOSITION OF ON PPMI MATRIX
# AND COMPUTATION OF COSINE SIMILARITY ON ROW VECTORS
dict_sorted = dm.SVD_cosine_matrix(Pmi_matrix, tokens, row_col)

#11. GENERATE ALL POSSIBLE COMBINATIONS OF QUERIES (QUERY EXPANSION)
final_queries = dm.query_expansion(tokens, dict_sorted, tokenizer)

#12. SHOW INFORMATON ABOUT EACH QUERY:
# QUERY - POSITION DOCUMENT TARGET - BEST SMOOTHING METHOD - SKIP-GRAM - PERPLEXITY
parameters = dm.show_information_queries(final_queries, query_obj, tokenizer)

#13. RETURN THE LIST OF QUERIES WITH THE LOWEST PERPLEXITY
dm.get_low_queries_perplexity(final_queries, parameters)




