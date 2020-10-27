import numpy as np
from scipy.sparse import csr_matrix, vstack
from time import time
from Q6_utils import *
import matplotlib.pyplot as plt



# Step0: Set hyper-parameters
LAMBDA = 1.0
NUM_ITER = 10
LEARNING_RATE = 0.00001
NUM_CLASS = 16
np.random.seed(93106)



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



# train_X = train_X[:10,:D].reshape((10,D))
# train_Y = train_Y[:10].reshape((10,))
# train_Y = np.array([1]*100).reshape((100,))
# train_X = np.random.normal(scale=0.1, size=D*3000).reshape((3000,D))
print(train_X.shape)
print(train_Y.shape)

train_X = train_X.toarray()
test_X = test_X.toarray()
train_X = (train_X - np.mean(train_X))/np.std(train_X)
test_X = (test_X - np.mean(train_X))/np.std(train_X)

# Step2: Initialize the parameter Theta
Theta = np.zeros([train_X.shape[1], NUM_CLASS])

# np.random.normal(scale=10e-5, size=D*15).reshape((D,15))
# Theta =  np.random.rand(10000,15)

# Step3: Train the model with stochastic gradient
train_acc_collection_sgd = []
test_acc_collection_sgd = []
time_training_collection_sgd = []
time_eclipse = 0.0
N = train_X.shape[0]
T = time()
for iter in range(NUM_ITER * N):
    Theta = train(Theta, train_X, train_Y, stochastic_gradient, LEARNING_RATE, 1, LAMBDA)
     
    if iter % N == 0:
        time_eclipse += time() - T
        train_acc = evaluation(Theta, train_X, train_Y)
        test_acc = evaluation(Theta, test_X, test_Y)

        # print("type(train_acc): ", type(train_acc))
        print(f"Iter {iter} train_acc: {train_acc:0.6f} , Test_error: {test_acc: 0.6f}")

        train_acc_collection_sgd.append(1.0 - train_acc)
        test_acc_collection_sgd.append(1.0 - test_acc)
        time_training_collection_sgd.append(time_eclipse)
        T = time()

# Step4: Train the model with full gradient
train_acc_collection_gd = []
test_acc_collection_gd = []
time_training_collection_gd = []
time_eclipse = 0.0
N = train_X.shape[0]
T = time()
for iter in range(NUM_ITER):
    Theta = train(Theta, train_X, train_Y, full_gradient, LEARNING_RATE, 1, LAMBDA)
    time_eclipse += time() - T
    train_acc = evaluation(Theta, train_X, train_Y)
    test_acc = evaluation(Theta, test_X, test_Y)

    # print("type(train_acc): ", type(train_acc))
    print(f"Iter {iter} train_acc: {train_acc:0.6f} , Test_error: {test_acc: 0.6f}")

    train_acc_collection_gd.append(1.0 - train_acc)
    test_acc_collection_gd.append(1.0 - test_acc)
    time_training_collection_gd.append(time_eclipse)
    T = time()

# plot
plt.figure()
plt.plot(time_training_collection_sgd, train_acc_collection_sgd, label='SGD-training error rate')
plt.plot(time_training_collection_sgd, test_acc_collection_sgd, label='SGD-testing error rate')
plt.plot(time_training_collection_gd, train_acc_collection_gd, label='GD-training error rate')
plt.plot(time_training_collection_gd, test_acc_collection_gd, label='GD-testing error rate')
plt.legend()
plt.xlabel('Training time')
plt.ylabel('Error rate')
plt.savefig('Q6e.png')
plt.show()