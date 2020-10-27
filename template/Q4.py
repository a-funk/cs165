import numpy as np
from scipy.sparse import csr_matrix
from math import log
from Q2c import generate_word_collection
import re
import math

from collections import OrderedDict

# define your global variables for question 2(b)
G_IDF = dict() # (optional) replace it with the global variables you use
num_docs = 0

word_collection = generate_word_collection('training.txt')


# Question 2(a)

def bag_of_word_feature(sentence):
    """
    @input:
        sentence: a string. Stands for "D" in the problem description. 
                            One example is "wish for solitude he was twenty years of age ".
    @output:
        encoded_array: a sparse vector based on library scipy.sparse.csr_matrix.
    """
    array = list()
    global word_collection
    word_collection = sorted(word_collection)
    word_split = sentence.strip().split()
    for word in word_collection:
        # count of word in sentence
        count = 0
        for split in word_split:
            if str(split) == str(word):
                count += 1
        array.append(float(count))

        # array.append(float(count))
        # print(count)

    encoded_array = csr_matrix(array)  # TODO
    return encoded_array


# Question 2(b)
def get_TF(term, document=None):
    """
    @input:
        term: str or list of str. words (e.g., [cat, dog, fish, are, happy])
        document: list of str. a sentence (e.g., ["wish", "for", "solitude").
            None if identical to term
    @output:
        TF: some datastructure containing frequency of each term in the document.
    """
    TF = dict()
    if isinstance(term, str):
        term = [term]
    if document is None:
        document = term
    total = len(document)
    for t in term:
        appear = term.count(t)
        TF[t] = appear/total
    return TF


def get_IDF(file_name):
    """
    @input:
        file_name: a string. should be either "training.txt" or "texting.txt"
    @output:
        None. Update the global variable you defined
    """
    global G_IDF  # (optional) replace it with the global variables you use
    # TODO
    global num_docs
    collection = generate_word_collection(file_name)
    collection = sorted(collection)
    with open(file_name, encoding='ISO-8859-1') as f:
        for line in f:
            num_docs += 1
            line = line.strip().split(',')[0] 
            # list of words in the document
            word = line.strip().split()
            wordset = set(word)
            for w in collection:
                if w in wordset:
                    if w in G_IDF:
                        G_IDF[w] = G_IDF[w] + 1
                    else:
                        G_IDF[w] = 1





def get_TF_IDF(term):
    """
    @input:
        term: str or list of str. words (e.g., [cat, dog, fish, are, happy])
    @output:
        TF_IDF: csr_matrix. Equal to TF*IDF.
    """
    arr = list()
    global G_IDF  # (optional) replace it with the global variables you use
    for t in word_collection:  # modify me if necessary
        try:
            #term is a list of the sentence
            if t in term:
                otc = G_IDF[t]
                IDF_value = math.log10(num_docs/otc)
                tf = get_TF(term)
                tfidf = tf[t] * IDF_value
                arr.append(tfidf)
            else:
                arr.append(0)
        except KeyError:
            pass  # if a word is not in the vocabulary, ignore it.
   
    return csr_matrix(arr)  # TODO


def TF_IDF_encoding(document):
    """
    @input:
        document: str. a sentence (e.g., "wish for solitude he was twenty years of age ").
    @output:
        encoded_array: a sparse vector based on library scipy.sparse.csr_matrix.
        Contain the TF_IDF_encoding of the given document.
    """
    global IDF

    return get_TF_IDF(document.split())


get_IDF('training.txt')

if __name__ == '__main__':
    word_collection = generate_word_collection("training.txt")
    sentence = "wish for solitude he was twenty years of age "
  

"""
Sample output:
    (0, 223)	1.0
    (0, 3497)	1.0
    (0, 4086)	1.0
    (0, 5975)	1.0
    (0, 8141)	1.0
    (0, 9234)	1.0
    (0, 9623)	1.0
    (0, 9811)	1.0
    (0, 9963)	1.0
The index can be different. It's totally fine.
"""

if __name__ == '__main__':
    sentence = "wish for solitude he was twenty years of age "
    #listSen = sentence.split()
    #print(get_TF_IDF(listSen))
    # sentence = "wish for for for solitude he he he was twenty years of age "
    print(TF_IDF_encoding(sentence))
    #print("num doc = ", num_docs)
    #print(get_TF)
    #print("IDF age = ", G_IDF["age"])
    # print(bag_of_word_feature(sentence))
    #print(G_IDF)

"""
Sample output:
  (0, 5)        0.00011272643105233294
  (0, 240)      4.8279086001969967e-05
  (0, 417)      0.002849094399422293
  (0, 607)      0.003883907659438026
  (0, 1019)     0.04815179197233144
  (0, 1422)     0.08039313286963955
  (0, 2342)     0.0893444830656407
  (0, 2462)     0.0908005616277569
  (0, 4205)     0.17253799651753118
The index can be different. It's totally fine.
"""
