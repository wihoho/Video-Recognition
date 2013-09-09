__author__ = 'GongLi'

import Utility as util
from sklearn.svm import SVC
import math
import numpy as np


def KFoldEvaluation(K):

    distances = util.loadObject("Data/Kodak_distanceMatrix_version2.pkl")
    labels = util.loadObject("Data/Kodak_labels_version2.pkl")

    # distances = util.loadObject("Data/KodakDistanceLevelOne.pkl")
    # labels = util.loadObject("Data/KodakLabelLevelOne.pkl")
    percentage = 1.0 / K

    # birthday: 0 - 15
    # parade: 16 - 29
    # picnic: 30 - 35
    # show: 36 - 92
    # sports: 93 - 167
    # wedding: 168 - 194

    birthdaIndex = 0
    paradeIndex = 16
    picnicIndex = 30
    showIndex = 36
    sportsIndex = 93
    weddingIndex = 168

    birthdayStep = int(percentage * 16)
    paradeStep = int(percentage * 14)
    picnicStep = int(percentage * 6)
    showStep = int(percentage * 57)
    sportsStep = int(percentage * 75)
    weddingStep = int(percentage * 27)

    for foldIndex in range(K):
        testIndex = []

        stop = birthdaIndex + birthdayStep
        if stop > 16:
            stop = 16
        testIndex += [i for i in range(birthdaIndex, stop, 1)]
        birthdaIndex = stop

        stop = paradeIndex + paradeStep
        if stop > 30:
            stop = 30
        testIndex += [i for i in range(paradeIndex, stop, 1)]
        paradeIndex = stop

        stop = picnicIndex + picnicStep
        if stop > 36:
            stop = 36
        testIndex += [i for i in range(picnicIndex, stop, 1)]
        picnicIndex = stop

        stop = showIndex + showStep
        if stop > 93:
            stop = 93
        testIndex += [i for i in range(showIndex, stop, 1)]
        showIndex = stop


        stop = sportsIndex + sportsStep
        if stop > 168:
            stop = 168
        testIndex += [i for i in range(sportsIndex, stop, 1)]
        sportsIndex = stop


        stop = weddingIndex + weddingStep
        if stop > 195:
            stop = 195
        testIndex += [i for i in range(weddingIndex, stop, 1)]
        weddingIndex = stop

        # construct training index
        # trainIndex = [i for i in range(195)] - testIndex
        trainIndex = [i for i in range(195)]
        for x in testIndex:
            trainIndex.remove(x)

        # construct trainDistance, testDistance
        trainSize = len(trainIndex)
        trainDistance = np.zeros((trainSize, trainSize))
        for i in range(trainSize):
            indexOne = trainIndex[i]
            for j in range(trainSize):
                indexTwo = trainIndex[j]
                trainDistance[i][j] = distances[indexOne][indexTwo]

        testSize = len(testIndex)
        testDistance = np.zeros((testSize, trainSize))
        for i in range(testSize):
            indexOne = testIndex[i]
            for j in range(trainSize):
                indexTwo = trainIndex[j]
                testDistance[i][j] = distances[indexOne][indexTwo]

        # construct testLabels, trainLabels
        testLabels = []
        for content in testIndex:
            testLabels.append(labels[content])

        trainLabels = []
        for content in trainIndex:
            trainLabels.append(labels[content])

        print "##############   "+ str(foldIndex)
        classification(trainDistance, testDistance, trainLabels, testLabels, "rbf")
        classification(trainDistance, testDistance, trainLabels, testLabels, "lap")
        classification(trainDistance, testDistance, trainLabels, testLabels, "id")
        classification(trainDistance, testDistance, trainLabels, testLabels, "isd")



def classification(trainDistance, testDistance, trainLabels, testLabels, kernelName):

    # more process on distance matrix
    trainDistance = trainDistance ** 2
    testDistance = testDistance ** 2

    meanTrainValue = np.mean(trainDistance)
    meanTestValue = meanTrainValue

    # meanTestValue = np.mean(testDistance)

    # ultimateMeaValue = (meanTestValue + meanTrainValue) / 2.0
    # meanTrainValue = ultimateMeaValue
    # meanTestValue = ultimateMeaValue

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


    decisionValues = clf.decision_function(testGramMatrix)


    SVMResults = clf.predict(testGramMatrix)
    correct = sum(1.0 * (SVMResults == testLabels))
    accuracy = correct / len(testLabels)
    print kernelName+ ": " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"
    # print str(accuracy)


if __name__ == "__main__":
    KFoldEvaluation(5)

    # print "--------------------rbf-------"
    # classifiyUsingDifferentKernel("rbf")
    # print " "
    #
    # print "--------------------lap-------"
    # classifiyUsingDifferentKernel("lap")
    # print " "
    #
    # print "--------------------id-------"
    # classifiyUsingDifferentKernel("id")
    # print " "
    #
    # print "--------------------isd-------"
    # classifiyUsingDifferentKernel("isd")
    # print " "







