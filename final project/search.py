#import nltk
import logging
from fasttext import ft_search
from tf_idf import tf_idf_search
from bm25 import bm25_search
from via_elmo import elmo_search, batcher, sentence_character_ids, \
    elmo_sentence_input
from flask import Flask
from flask import render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
#nltk.download("stopwords")

logging.basicConfig(filename="search.log", level=logging.DEBUG)
logging.info("Program started")
app = Flask(__name__)

with open("documents.pickle", "rb") as c:
    docs = pickle.load(c)

with open("corpus.pickle", "rb") as c:
    corpus = pickle.load(c)

N = len(corpus)

X = vectorizer.fit_transform(corpus)
f_matrix = X.toarray()
tf_matrix = np.transpose(f_matrix)

doc_words = vectorizer.get_feature_names()

all_n = {}
for word in doc_words:
    w_idx = doc_words.index(word)
    all_n[word] = np.count_nonzero(tf_matrix[w_idx])

@app.route('/')
def index_main():
    logging.info("Opening the main page")
    return render_template('search.html')


@app.route('/tf_idf_results')
def res_tf_idf():
    logging.info("User took the tf-idf model")
    query = request.args['search']
    logging.info("Processing request")
    result = tf_idf_search(query)
    response = []
    for i in result:
        response.append(str(i[0] + ', score = ' + str(i[1])))
    logging.info("The tf-idf initialized, rendering the page")
    return render_template('tf_idf_results.html', query=query,
                           response=response)


@app.route('/fasttext_results')
def res_fasttext():
    logging.info("User took the fasttext model")
    query = request.args['search']
    logging.info("Processing request")
    result = ft_search(query, docs)
    response = []
    for i in result:
        response.append(str(i[0] + ', score = ' + str(i[1])))
    logging.info("The fasttext initialized, rendering the page")
    return render_template('fasttext_results.html', query=query,
                           response=response)


@app.route('/bm25_results')
def res_bm25():
    logging.info("User took the bm25 model")
    query = request.args['search']
    logging.info("Processing request")
    result = bm25_search(query)
    response = []
    for i in result:
        response.append(str(i[0] + ', score = ' + str(i[1])))
    logging.info("The bm25 initialized, rendering the page")
    return render_template('bm25_results.html', query=query, response=response)


@app.route('/elmo_results')
def res_elmo():
    logging.info("User took the elmo model")
    query = request.args['search']
    logging.info("Processing request")
    result = elmo_search(query, batcher, sentence_character_ids,
                         elmo_sentence_input, docs)
    response = []
    for i in result:
        response.append(str(i[0] + ', score = ' + str(i[1])))
    logging.info("The elmo initialized, rendering the page")
    return render_template('elmo_results.html', query=query, response=response)


if __name__ == '__main__':
    app.run(debug=True)
