__author__ = 'GongLi'

import Utility as util
from sklearn.svm import SVC
import math

distances = util.loadDataFromFile("Data/distanceMatrix.pkl")
labels = util.loadDataFromFile("Data/labels.pkl")

# Training data
trainDistance = distances[0::2, 0::2]
trainLabels = labels[0::2]
trainGramMatrix = math.e ** (0 - trainDistance ** 2)

# Testing data
testDistance = distances[1::2, 0::2]
testLabels = labels[1::2]
testGramMatrix = math.e ** (0 - testDistance ** 2)

# SVM
clf = SVC(kernel="precomputed")
clf.fit(trainGramMatrix, trainLabels)
SVMResults = clf.predict(testGramMatrix)
correct = sum(1.0 * (SVMResults == testLabels))
accuracy = correct / len(testLabels)
print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"






