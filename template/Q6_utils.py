import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from math import log
from os import path
import pickle
from Q2c import generate_word_collection
from Q4a import bag_of_word_encoding
from Q4b import N_Gram
from Q4c import TF_IDF_encoding
from scipy.special import logsumexp

def log_softmax(x):
    """
    @input:
        x: a 2-dimension numpy array
    @output:
        result: a numpy array of the same shape as x
    Compute log softmax of x along second dimension.
    Using this function provides better numerical stability than using softmax
    when computing cross entropy loss
    """
    return x - logsumexp(x, axis=1, keepdims=True)


def get_one_hot(Y, k):
    if not isinstance(Y, np.ndarray) or len(Y.shape) == 0:
        Y = np.array([Y])
    b = Y.shape[0]
    return csr_matrix((np.ones(b), (np.arange(b), Y)), shape=[b, k])


# Question 6(a)
def inference(theta, x):
    """
    @input:
        theta: a numpy matrix of shape [d,k]
        x: a numpy.ndarray of shape [d] or [b, d]
            if you're not comfortable handing x as 2-d array, 
            you can assume x is a numpy.array of shape [d]
    @output:
        result: a numpy array of shape [b, k].
    """
    
    # delete me if x is always 1-d
    if len(x.shape) == 1:
        x = x.reshape(1, -1) # convert to shape [b, d]
    # TODO
    return 


# TODO


# Question 6(b)
def gradient(x, y, Theta):
    """
    gradient of cross-entropy loss with respect to Theta
    @input:
        x: a numpy array. size of (d,) or (b, d)
        y: a numpy array. size of (k,) or (b, k)
        Theta: a numpy array. size of (d,k)
        if you're not comfortable handing x and y as 2-d array, 
            you can assume x and y is a numpy.array of shape [d]
    @output:
        grad: gradient w.r.t. parameter Theta.
    """
    
    # delete me if x is always 1-d
    if len(x.shape) == 1:
        x = x.reshape(1, -1) # convert to shape [b, d]
    if len(y.shape) == 1:
        y = y.reshape(1, -1) # convert to shape [b, d]

    # TODO
    return 


# Question 6(c)
def full_gradient(Theta, X, Y, Lambda):
    """
    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (b,d)
        Y: a numpy array. size of (b,)
    @return:
        gradient_sum: a numpy array. size of (d,k). Full gradient to Theta, averaged over all data points. 
    """
    k = Theta.shape[1]
    Y_one_hot = get_one_hot(Y, k)
    # TODO
    return

def stochastic_gradient(Theta, X, Y, Lambda):
    """
    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (N,d)
        Y: a numpy array. size of (N,)
    @return:
        gradient_sum: a numpy array. size of (d,k). Stochastic gradient to Theta on a single sampled data point.
    """

    k = Theta.shape[1]
    b = X.shape[0]
    idx = np.random.randint(b)
    y = get_one_hot(Y[idx], k)
    x = X[idx, :]
    # TODO
    return 


# Question 6(d)
def train(Theta, X, Y, gradient_function, learning_rate, num_iter, Lambda):
    """
    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (N,d)
        Y: a numpy array. size of (N,)
        gradient_function:  a function. Should be either "full_gradient", "stochastic_gradient"
        learning_rate: a float.
        num_iter: an integer. Number of iterations.
        Lambda: a float. The regularization term.
    @return: 
        Update_Theta: a numpy array. size of (d,k). Updated parameters.
    """
    for _ in range(num_iter):
        gradient_update = gradient_function(Theta, X, Y, Lambda)
        # TODO
    return Theta    

# Question 6(e)


# Utility2: Evaluation Error
def evaluation(Theta, X, Y):
    """
    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (d,d)
        Y: a numpy array. size of (d,)
    @output:
        eval_acc: a float. evaluation accuracy.
    """

    Y_hat = inference(Theta, X)
    Y_argmax = np.argmax(Y_hat, axis=1)
    eval_acc = float(np.sum(Y_argmax == Y))/len(Y)
    return eval_acc

# Utility1: Data preparation based on TF-IDF.
def data_prepare(file_name, word_collection):
    """
    @input:
        file_name: a string. should be either "training.txt" or "texting.txt"
        word_collection: a list. Refer to the output of generate_word_collection(file_name).
    @return:
        X: a numpy array. size of (N,d)
        Y: a numpy array. size of (N,)
    """
    X = []
    Y = []
    with open(file_name, encoding='ISO-8859-1') as f:
        for line in f:
            input_string, y = line.strip().split(',')
            x = TF_IDF_encoding(input_string, file_name)
            X.append(x)
            Y.append(int(y))
    X = vstack(X)
    Y = np.array(Y)
    return X, Y