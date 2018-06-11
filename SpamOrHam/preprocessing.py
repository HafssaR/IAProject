import os
from nltk.corpus import stopwords
from collections import Counter

Training_data='D:/IAProjet/ProjetIA/TrainSet'
def vocab_list(directory_path):
    wordList = [] 
    emails = [os.path.join(directory_path,file) for file in os.listdir(directory_path)]     
    for email in emails:    
        with open(email) as content:
            for i,line in enumerate(content):
                    #Lowercasing the content of the email
                    line.lower()
                    #spliting the email into words to constitute the vocabulary list
                    words = line.split()
                    wordList += words
    # subclass list for counting hashable objects (words)
    #Value of each key is the occurrence of the words in the email
    vocab_list = Counter(wordList)
    stop=stopwords.words('english')
    # code for non-word removal 
    keyList = vocab_list.keys()
    for i in list(vocab_list) :
        if i.isalpha() == False: 
            #removing irrelevant numbers
            del vocab_list[i]
        elif len(i) == 1:
            #removing irrelevant single characters
            del vocab_list[i]
            #removing irrelevant words non indentifiying the spams
        elif i in stop:
            del vocab_list[i]
    
    #dict= stopwords.words('english')
    #using most_common in order to sort the items of the vocabulary list
    #4000 most common words used in preprocessing the emails
    #it returns the key and its value
    vocab_list = vocab_list.most_common(1300)
    return vocab_list

