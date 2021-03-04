import spacy
import nltk.stem.snowball
# uncomment if and only if is the first execution
#nltk.download('punkt')
#nltk.download('wordnet')
import snowballstemmer
import csv
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

stemmer = snowballstemmer.stemmer('english')
nlp = spacy.load('en_core_web_sm')

mac_path = '/Users/flavioforenza/Desktop/Information-Retrieval-project/ricette.csv'

with open(mac_path, 'r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        # scorre ogni riga
        row = ''.join(row)
        row = row.replace(';', '').replace('**', '').replace(',', '').replace('(', '').replace(')', '')

        # lista  contenente i token di ogni ogni frase (per ogni punto trovato)
        fs = sent_tokenize(row)

        # 1 token = 1 stringa (frase)
        tokens = nlp(row)

        # lista che contiene tutte le parole tokenizzare di 1 frase
        words = ['#i'] + [x.text for x in tokens] + ['#f']

        # lista ceh conterrà tutti i token (dopo la rimozione delle Stopwords)
        filtered_sentence = []

        for word in words:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word)

        # ******** LEMMATIZZAZIONE ********

        # lista che conterrà i lemmi (con gensim utilizzando Wordnet)
        lemma_word = []
        wordnet_lemmatizer = WordNetLemmatizer()
        for w in filtered_sentence:
            # nome
            word1 = wordnet_lemmatizer.lemmatize(w, pos="n")
            # verbo
            word2 = wordnet_lemmatizer.lemmatize(word1, pos="v")
            # aggettivo
            word3 = wordnet_lemmatizer.lemmatize(word2, pos=("a"))
            lemma_word.append(word3)

        print(lemma_word)