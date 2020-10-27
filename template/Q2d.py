def factorial(n):
    """
    @input:
        n: a positive integer. 
    @output:
        an integer of n!.
    """
    if n == 1 or n==0:
        return n
    else:
        return factorial(n-1)*n
    # TODO

if __name__ == '__main__':
    print(factorial(3))

"""
Output example:
6
"""

