import numpy as np
from scipy.sparse import csr_matrix, vstack
from Q6_utils import *
import matplotlib.pyplot as plt





# Step0: Set hyper-parameters
LAMBDA = [0.00001,0.0001,0.001,0.01, 0.1,1,10,100,1000]
NUM_ITER = 10
LEARNING_RATE = 0.00001
NUM_CLASS = 16




# Step1: Preprocess X, Y. Prepare for training.
if path.exists("word_collection.pickle"):
    print("load word_collction")
    word_collection = pickle.load(open("word_collection.pickle",'rb'))
else:
    print("compute word_collction")
    word_collection = generate_word_collection('training.txt')
    pickle.dump(word_collection, open("word_collection.pickle",'wb'))


if path.exists('train_X.pickle'):
    print("load train_X and train_Y")
    train_X = pickle.load(open("train_X.pickle", 'rb'))  
    train_Y = pickle.load(open("train_Y.pickle", 'rb'))  
else:
    print("compute train_X and train_Y")
    train_X, train_Y = data_prepare('training.txt', word_collection)
    pickle.dump(train_X, open("train_X.pickle", 'wb'))
    pickle.dump(train_Y, open("train_Y.pickle", 'wb'))

if path.exists("test_X.pickle"):
    print("load test_X and test_Y")
    test_X = pickle.load(open("test_X.pickle", 'rb'))
    test_Y = pickle.load(open("test_Y.pickle", "rb"))
else:
    print("compute test_X and test_Y")
    test_X, test_Y = data_prepare('testing.txt', word_collection)
    pickle.dump(test_X, open("test_X.pickle", 'wb'))
    pickle.dump(test_Y, open("test_Y.pickle", 'wb'))


train_X = train_X.toarray()
test_X = test_X.toarray()
train_X = (train_X - np.mean(train_X))/np.std(train_X)
test_X = (test_X - np.mean(train_X))/np.std(train_X)

train_acc_collection = []
test_acc_collection = []

for lamb in LAMBDA:
    np.random.seed(93106)
    # Step2: Initialize the parameter Theta
    Theta = np.zeros([train_X.shape[1], NUM_CLASS])

    
    time_training_collection_sgd = []
    N = train_X.shape[0]
    for iter in range(NUM_ITER * N):
        Theta = train(Theta, train_X, train_Y, stochastic_gradient, LEARNING_RATE, 1, lamb)
        
        
            
    train_acc = evaluation(Theta, train_X, train_Y)
    test_acc = evaluation(Theta, test_X, test_Y)

    # print("type(train_acc): ", type(train_acc))
    print(f"Train_acc: {train_acc:0.6f} , Test_error: {test_acc: 0.6f}")

    train_acc_collection.append(1.0 - train_acc)
    test_acc_collection.append(1.0 - test_acc)


    

# plot
plt.figure()
plt.loglog(LAMBDA, train_acc_collection, label='SGD-training accuracy')
plt.loglog(LAMBDA, test_acc_collection, label='SGD-testing accuracy')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error rate')
plt.savefig('Q6e.png')
plt.show()