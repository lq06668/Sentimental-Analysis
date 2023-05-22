from __future__ import unicode_literals
import csv
import pandas as pd
import numpy as np
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import string
import codecs
import glob
from collections import Counter
import re
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from pandas import *
import xlrd

################## TRAINING AND TESTING OF MODEL FOR ACCURACY #####################


def load_file(fileName):
    data = pd.read_excel(fileName)
    return data


def count_words(data):
    Sent = []
    # sentences = sentences.translate(table)
    for sentences in data:
        sentences = str(sentences)
        sentences = re.sub(r"\d+", " ", sentences)
        # English punctuations
        sentences = re.sub(
            r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]+""", " ", sentences)
        # Urdu punctuations
        sentences = re.sub(r"[:؛؟’‘٭ء،۔]+", " ", sentences)
        # Arabic numbers
        sentences = re.sub(r"[٠‎١‎٢‎٣‎٤‎٥‎٦‎٧‎٨‎٩]+", " ", sentences)
        sentences = re.sub(r"[^\w\s]", " ", sentences)
        # Remove English characters and numbers.
        sentences = re.sub(r"[a-zA-z0-9]+", " ", sentences)
        # remove multiple spaces.
        sentences = re.sub(r" +", " ", sentences)
        Sent.append(sentences)
        words = sentences.split()
        #print("words: ", words)
        counter = Counter(words)
    #print("Sentences:", Sent)
    vectorizer = CountVectorizer()
    vectorizer.fit(Sent)
    vector = vectorizer.transform(Sent)
    #print("Vocabulary: ", vectorizer.vocabulary_)
    print("2D array:")
    y = vector.toarray()
    print(vector.toarray())
    return vector


def learn_model(data, target):

    classifier = None
    # Your custom implementation of NaiveBayes classifier will go here.
    print("target", target)
    count = -1
    # print(data)
    print("News")
    category = list(np.unique(target))
    for i in category:
        count = count+1
        if i == 3:
            category.pop(count)
    #category = [0, 1, 2]
    print("Category:", category)
    nestedlist = data.toarray()  # coverted to array of vectors
    # print(nestedlist)
    rowcol = nestedlist.shape
    row = rowcol[0]
    col = rowcol[1]
    initialize = np.zeros((len(category), col))
    # since we have 3 categories so each index of each category
    classprob = [0, 0, 0]
    # print(initialize)
    for k in range(len(category)):
        TwoDArray = nestedlist[category[k] == target]
        rowcol2 = TwoDArray.shape
        row2 = rowcol2[0]
        classprob[k] = row2/row
        neww = numpy.sum(TwoDArray, axis=0)
        initialize[k] = neww
        count = 0
        for i in initialize[k]:
            # part b here of Laplacian smoothing to avoid 0 probabilities
            initialize[k][count] = i+1
            count += 1
        neww2 = numpy.sum(initialize[k])
        # this gives us conditional probability of each class.
        initialize[k] = np.divide(initialize[k], neww2)
    final = [classprob, initialize]
    classifier = final
    return classifier


def classify(classifier, testdata):

    predicted_val = []
    testingarray = testdata.toarray()
    #testingarray = np.array(testdata)
    #print("testing arr", testingarray)
    classprob = classifier[0]
    conditionalprob = classifier[1]
    rowcol = testingarray.shape
    rows = rowcol[0]
    for i in range(rows):
        problist = []
        for j in range(3):
            mult = 1
            mult = mult*classprob[j]
            for k in range(len(testingarray[i])):
                if testingarray[i][k] >= 1:
                    mult = mult*conditionalprob[j][k]
            problist.append(mult)
        final = max(problist)
        index = problist.index(final)
        if index == 0:
            predicted_val.append(0)
        elif (index == 1):
            predicted_val.append(1)
        else:
            predicted_val.append(2)

    # print(predicted_val)
    return predicted_val


def evaluate(actual_class, predicted_class):

    count = -1
    for i in actual_class:
        count = count+1
        if i == 3:
            actual_class.pop(count)

    print("actual", actual_class)
    accuracy = precision = recall = f_measure = -1
    # print(actual_class)
    # row is predicted and columns are actual
    confusion_matrix = np.zeros((3, 3))
    count = 0
    for j in actual_class:
        if predicted_class[count] == 0:
            row = 0
            count += 1
        elif predicted_class[count] == 1:
            row = 1
            count += 1
        elif predicted_class[count] == 2:
            row = 2
            count += 1
        if j == 0:
            col = 0
        elif j == 1:
            col = 1
        elif j == 2:
            col = 2
        confusion_matrix[row][col] += 1
    divisor = np.sum(confusion_matrix)
    accuracy = np.diag(confusion_matrix).sum()/divisor
    precisionlist = []
    recalllist = []
    f_list = []
    for i in range(confusion_matrix.shape[0]):
        TP = confusion_matrix[i, [i]]
        FP = confusion_matrix[[i], :].sum()-TP
        FN = confusion_matrix[:, [i]].sum()-TP
        TN = confusion_matrix.sum().sum() - TP-FP-FN
        if (TP+FP == 0):
            precisionlist.append(TP+FP)
            recalllist.append(TP+FP)
            f_list.append(TP+FP)
        else:
            precision = TP/(TP+FP)
            precisionlist.append(precision)
            recall = TP/(TP+FN)
            recalllist.append(recall)
            f_measure = (2*TP)/((2*TP)+FN+FP)
            f_list.append(f_measure)

    print("Accuracy:", accuracy)
    print("          Positive ", "       Negative ", "      Neutral ")
    print("Precision:", precisionlist)
    print("Recall:   ", recalllist)
    print("F-measure:", f_list)
    print("Confusion Matrix: \n", confusion_matrix)
    # Write your code to print confusion matrix here
    # references: https://www.youtube.com/watch?v=zqJLLaDd4QQ


def data_frequency(dataset):
    y = list(dataset["Sentiment"].value_counts())
    plt.plot()
    plt.bar(["Positive", "Negative", "Neutral"], y)
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.title("Sentiment frequency of Dataset")
    plt.show()

    print("Positive {}, Negative {}, Neutral {}".format(y[0], y[1], y[2]))


print("Data loading")
dataset = load_file(r"C:\Users\LAIBA\Desktop\AI project final\new_data.xlsx")
data_frequency(dataset)
data, target = dataset['Text'].tolist(), dataset['Sentiment'].tolist()


word_vectors = count_words(data)

trainingX, testX, trainingY, testY = train_test_split(
    word_vectors, target, test_size=0.4, random_state=43)


print("Learning model.....")
model = learn_model(trainingX, trainingY)

print("Classifying test data......")


predictedY = classify(model, testX)

#### PREDICTED VALUES OF YOUR INPUT #####
print(predictedY)

print("Evaluating results.....")
evaluate(testY, predictedY)
