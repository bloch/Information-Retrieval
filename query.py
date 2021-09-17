import json
import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def query(path, question):
    with open(path) as f:
        data = json.load(f)
    D = data["D"]

    question_tokens = nltk.word_tokenize(question)
    question_tokens = [ps.stem(token) for token in question_tokens if token.isalpha() and token.lower() not in stop_words]
    query_counter = dict(Counter(question_tokens))
    max_counter = Counter(question_tokens).most_common(1)[0][1]

    R = [[i+1, 0.0] for i in range(D)]
    L = 0.0
    for token, count in query_counter.items():
        if token in data["words"]:
            I = data["words"][token]["idf_i"]
            K = count/max_counter
            W = K*I
            L += math.pow(W, 2)
            list = data["words"][token]["list"]
            for inner_hash in list:
                D = inner_hash["doc_id"]
                C = inner_hash["tf"]

                R[D-1][1] += W*I*C

    L = math.sqrt(L)
    for i in range(D):
        S = R[i][1]    #document (i+1)
        Y = data["lengths"][str(i+1)]
        R[i][1] = S/(Y*L)

    R = [tup for tup in R if tup[1] >= 0.09]
    R.sort(reverse=True, key=lambda tup: tup[1])

    return R
