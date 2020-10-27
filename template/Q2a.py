import numpy as np


#Note: Should use numpy array for A and B


# A: 5x4
# B: 4x3

A = np.arange(1,21,1).reshape((5,4))
print(A)


B = np.arange(1,13,1).reshape((3,4)).T
print(B)

AxB = np.dot(A,B)
print(AxB)


"""
Output example:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]]
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
[[ 70  80  90]
 [158 184 210]
 [246 288 330]
 [334 392 450]
 [422 496 570]]
"""
