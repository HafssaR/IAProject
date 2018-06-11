import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
# Training classifier


Training_labels = np.zeros(4000)
 #986 Spams within 3498 emails for training the classifier
 #spam label equals 1 non spam equals 0
Training_labels[1831:4000] = 1
SVM_Classifier = LinearSVC()    #Support Vector Machines (SVM)
SVM_Classifier.fit(Training_sets,Training_labels)

# Testing classifier
Testing_labels = np.zeros(1677)
#336 Spams within 1477 emails for testing
Testing_labels[731:1677] = 1 

def Prediction(classifier,test_set):
    prediction= classifier.predict(test_set)
    return prediction

def Accuracy_Evaluation(test_target,prediction):
#quantifying the quality of predictions 
    return ('Evaluating the Accuracy = ' + str(accuracy_score(test_target,prediction)))
  
pred=Prediction(SVM_Classifier,Testing_sets)
print (pred)
#recall and precision
print (Accuracy_Evaluation(Testing_labels,pred))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
confusion_matrix=confusion_matrix(Testing_labels,pred)
print (confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Matrice de confusion')
plt.colorbar()
plt.ylabel('Etiquettes de tests reels')
plt.xlabel('Etiquettes de tests de prediction')
plt.show()
