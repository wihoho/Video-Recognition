__author__ = 'GongLi'

from pulp import *
import numpy as np
import os
import Utility as util
import subprocess as sub

def EMD(feature1, feature2, w1, w2):
    os.environ['PATH'] += os.pathsep + '/usr/local/bin'

    H = feature1.shape[0]
    I = feature2.shape[0]

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

    # Set variables for EMD calculations
    variablesList = []
    for i in range(H):
        tempList = []
        for j in range(I):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(H):
        for j in range(I):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += lpSum(objectiveFunction)


    tempMin = min(sum(w1), sum(w2))
    problem += lpSum(constraint) == tempMin

    # constraints
    for i in range(H):
        constraint1 = [variablesList[i][j] for j in range(I)]
        problem += lpSum(constraint1) <= w1[i]

    for j in range(I):
        constraint2 = [variablesList[i][j] for i in range(H)]
        problem += lpSum(constraint2) <= w2[j]

    # solve
    problem.writeLP("EMD.lp")
    problem.solve(GLPK_CMD())

    flow = value(problem.objective)

    return flow / tempMin

def C_EMD(feature1, feature2):
    # os.environ['PATH'] += os.pathsep + '/usr/local/bin'

    if feature1.shape[0] > 349:
        feature1 = feature1[:350]
    if feature2.shape[0] > 349:
        feature2 = feature2[:350]

    H = feature1.shape[0]
    I = feature2.shape[0]

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])


    groundDistanceFile = open("groundDistance", "w")
    groundDistanceFile.write(str(H) +" "+ str(I) +"\n")

    distances = distances.reshape((H * I, 1))
    for i in range(H * I):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    # os.system("/Users/GongLi/PycharmProjects/VideoRecognition/EarthMoverDistance")
    sub.call(["/Users/GongLi/PycharmProjects/VideoRecognition/EarthMoverDistance"])

    # Read in EMD distance
    file = open("result", "r").readlines()
    # os.remove("groundDistance")

    return float(file[0])

def calculaeDistanceMatrix(P):

    labels = []
    videoData = []


    for item in os.listdir(P):
        if item in [".DS_Store", "voc.pkl", "distanceMatrix.pkl", "labels.pkl"]:
            continue

        classPath = P +"/"+ item

        for video in os.listdir(classPath):
            completePath = classPath + "/" +video
            print completePath

            videoHistogram = util.loadObject(completePath)
            videoData.append(videoHistogram)

            labels.append(item)

    del videoHistogram
    videoHistogram = None

    # Calculate the distance matrix
    numberOfVideos = len(labels)
    distanceMatrix = np.zeros((numberOfVideos, numberOfVideos))

    for i in range(0, numberOfVideos):
        for j in range(i, numberOfVideos, 1):
            if i == j:
                distanceMatrix[i][j] = 0
                continue

            image1 = videoData[i]
            image2 = videoData[j]

            if image1.shape[0] > 349:
                # sampleRate = int(image1.shape[0] / 100) + 1
                image1 = image1[:350]

            if image2.shape[0] > 349:
                # sampleRate = int(image2.shape[0] / 100) + 1
                image2 = image2[:350]

            distanceMatrix[i][j] = C_EMD(image1, image2)
            distanceMatrix[j][i] = distanceMatrix[i][j]

            print "["+str(i) +","+ str(j)+"]: " + str(distanceMatrix[i][j])

    return distanceMatrix, labels

if __name__ == "__main__":
    distanceMatrix, labels = calculaeDistanceMatrix("KodakHistograms")
    util.writeDataToFile("Data/Kodak_distanceMatrix_version2.pkl", distanceMatrix)
    util.writeDataToFile("Data/Kodak_labels_version2.pkl", labels)