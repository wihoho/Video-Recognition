__author__ = 'GongLi'

import Utility as util
from sklearn.svm import SVC
import math
import numpy as np

def classifiyUsingDifferentKernel(kernelName):

    distances = util.loadDataFromFile("Data/distanceMatrix.pkl")
    labels = util.loadDataFromFile("Data/labels.pkl")

    # Training data
    trainDistance = distances[0::2, 0::2]
    trainLabels = labels[0::2]

    # Testing data
    testDistance = distances[1::2, 0::2]
    testLabels = labels[1::2]

    # more process on distance matrix
    trainDistance = trainDistance ** 2
    testDistance = testDistance ** 2

    meanTrainValue = np.mean(trainDistance)
    meanTestValue = np.mean(testDistance)

    # Different gram matrix with different kernels
    if kernelName == "rbf":
        trainGramMatrix = math.e ** (0 - trainDistance / meanTrainValue)
        testGramMatrix = math.e ** (0 - testDistance / meanTestValue)

    elif kernelName == "lap":
        trainGramMatrix = math.e ** (0 - (trainDistance / meanTrainValue) ** (0.5))
        testGramMatrix = math.e ** (0 - (testDistance / meanTestValue) ** (0.5))
    elif kernelName == "id":
        trainGramMatrix = 1.0 / ((trainDistance / meanTrainValue) ** (0.5) + 1.0)
        testGramMatrix = 1.0 / ((testDistance / meanTestValue) ** (0.5) + 1.0)
    elif kernelName == "isd":
        trainGramMatrix = 1.0 / (trainDistance / meanTrainValue + 1)
        testGramMatrix = 1.0 / (testDistance / meanTestValue + 1)

    # SVM
    clf = SVC(kernel="precomputed")
    clf.fit(trainGramMatrix, trainLabels)
    SVMResults = clf.predict(testGramMatrix)
    correct = sum(1.0 * (SVMResults == testLabels))
    accuracy = correct / len(testLabels)
    print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

if __name__ == "__main__":

    print "--------------------rbf-------"
    classifiyUsingDifferentKernel("rbf")
    print " "

    print "--------------------lap-------"
    classifiyUsingDifferentKernel("lap")
    print " "

    print "--------------------id-------"
    classifiyUsingDifferentKernel("id")
    print " "

    print "--------------------isd-------"
    classifiyUsingDifferentKernel("isd")
    print " "






