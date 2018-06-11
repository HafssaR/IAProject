import os
import numpy as np

Training_data='D:/IAProjet/ProjetIA/TrainSet'
dic=vocab_list(Training_data)

def Processing_email(directory_path, dic): 
    emails = [os.path.join(directory_path,file) for file in os.listdir(directory_path)]
    X= np.zeros((len(emails),len(dic)))
    email_id = 0
    word_id = 0
    for email in emails:
      with open(email) as content:
        for i,line in enumerate(content):
            line.lower()
            words = line.split()
            for word in words:
              for j,d in enumerate(dic):
                if d[0] == word:
                  word_id = j
                  X[email_id,word_id] = words.count(word)
        email_id += email_id     
    return X

Training_sets=Processing_email('D:/IAProjet/ProjetIA/TrainSet', dic )
Testing_sets=Processing_email('D:/IAProjet/ProjetIA/TestSet', dic)
