import numpy as np


# Hint: You may want to use "ISO-8859-1" encoding in open(filename, encoding="ISO-8859-1").

def generate_word_collection(file_name):
    '''
    @input:
        file_name: a string. should be either "training.txt" or "testing.txt"
    @return:
        word_collection: use the data structure you find proper to repesent the words 
        as well as how many times the word appears in a given file.
    '''
   # O(n^2) I think.  Reads each word once, but checks each word against the keys of word_list
   # But in reality its much closer to O(n) because the number of keys (the number of unique words)
   # in word_list is likey far smaller than n since you presume many words repeat themselves many 
   # times. In fact the max number of keys to be compared against per word is 0.3% of n.    
    word_list={}
    n=0
    with open(file_name,'r',  encoding='ISO-8859-1') as file: 
   
        # reading each line     
        for line in file: 

            # reading each word         
            line = line.strip().split(',')[0] 
            words = line.strip().split()
            for word in words:
                n=n+1
                if word not in word_list.keys():
                    word_list[word] = 1
                else:   
                    word_list[word] = word_list[word] + 1
    # print("Number of words "+str(n))
    # print("Number of unique words " + str(len(word_list.keys())))
    # print(len(word_list.keys())/n)
    return word_list




def print_top_k(word_collection, k):
    '''
    @input:
        word_collection: output of generate_word_collection
        k: a int. Indicate the top-k words to print. Should be 20 in question Q2(c).
    @return:
        None. Result is printed.
    '''
    # Step1: Sort all word in word_collection based on its count, from large to small.
    sort_words = sorted(word_collection.items(), key=lambda x: x[1], reverse=True)
    # Step2: Print the first k elements in the sorted word_collection. 
    i = 0
    for item in sort_words:
        if(i==k):
            return 0
        print(item)
        i = i+1

if __name__ == '__main__':
    word_collection = generate_word_collection("training.txt")
    print_top_k(word_collection, 20)

"""
Output example:
('the', 190806)
('of', 116447)
('and', 102422)
('to', 89251)
('a', 72558)
('in', 56028)
('i', 45308)
('that', 39378)
('he', 38160)
('it', 34361)
('was', 33834)
('his', 28771)
('â', 28082)
('with', 25176)
('as', 25032)
('for', 23788)
('is', 22857)
('you', 22808)
('her', 21188)
('had', 20869)
"""