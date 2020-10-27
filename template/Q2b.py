import numpy as np
from scipy.sparse import csr_matrix

# Note: Should use scipy.sparse library.

def generate(n):
    """
    @input:
        n: an int. 
    @return:
        csr_matrix
    """
    assert n > 1
    
    A = np.zeros(shape=(n-1,n))
    

    for i in range(n):
        for j in range(n-1):
            if(i==j):
                A[i][j] = 1
                A[i][j+1] = -1

    A_s = csr_matrix(A)
    #A = #TODO
    return A_s

A = generate(5)
print(A)

# for small n, also try running "print(A.todense())", this may help you to debug

"""
Output example:
    (0, 0)	1.0
    (0, 1)	-1.0
    (1, 1)	1.0
    (1, 2)	-1.0
    (2, 2)	1.0
    (2, 3)	-1.0
    (3, 3)	1.0
    (3, 4)	-1.0
"""
