class Query:

    def __init__(self, idx_q, categories, id_doc, query):
        self.index = idx_q
        self.categories = categories
        self.id_doc = id_doc
        self.query = query

    def set_ranking(self, doc_score):
        self.doc_score = doc_score

    def set_threshold(self, threshold):
        self.threshold = threshold

    def relevant_ranking_tfidf(self):
        return [k[0] for k, i in self.doc_score if i >= self.threshold]


