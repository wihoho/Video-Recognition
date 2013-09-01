__author__ = 'GongLi'

import calculateDistanceMatrix as calDis
import numpy as np
import os
import Utility as util
import subprocess as sub

def calculateDistanceAtLevelOne(videoOne, videoTwo):
    assert len(videoOne) == 8
    assert len(videoTwo) == 8

    # calculate volume to volume distance
    distances = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):

            distances[i][j] = calDis.C_EMD(videoOne[i], videoTwo[j])

    # calculate aligned distance
    return alignedDistances(distances)

def alignedDistances(distances):
    # os.environ['PATH'] += os.pathsep + '/usr/local/bin'
    shape = distances.shape
    assert shape[0] == 8 and shape[1] == 8

    # define linear programming
    groundDistanceFile = open("groundDistance", "w")
    groundDistanceFile.write("8" +" "+ "8" +"\n")

    distances = distances.reshape((64, 1))
    for i in range(64):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    sub.call(["/Users/GongLi/PycharmProjects/VideoRecognition/EarthMoverDistance"])

    # Read in EMD distance
    file = open("result", "r").readlines()
    os.remove("groundDistance")

    return float(file[0])


def calculateDistanceMatrixAtLevelOne(PATH):
    labels = []
    videoData = []

    for domain in os.listdir(PATH):
        if domain == ".DS_Store":
            continue

        P = PATH +"/"+ domain
        print P

        for video in os.listdir(P):
            if video == ".DS_Store":
                continue

            completePath = P +"/"+ video
            print completePath

            videoHistograms = util.loadObject(completePath)
            videoData.append(videoHistograms)

            labels.append(domain)

    numberOfVideos = len(labels)
    distanceMatrix = np.zeros((numberOfVideos, numberOfVideos))

    for i in range(numberOfVideos):
        for j in range(i, numberOfVideos, 1):

            if i == j:
                distanceMatrix[i][j] = 0
                continue

            distanceMatrix[i][j] = calculateDistanceAtLevelOne(videoData[i], videoData[j])
            distanceMatrix[j][i] = distanceMatrix[i][j]

            print "["+str(i) +","+ str(j)+"]: " + str(distanceMatrix[i][j])

    return distanceMatrix, labels

if __name__ == "__main__":

    distanceMatrix, labels = calculateDistanceMatrixAtLevelOne("KodakLevelOneHistograms")
    util.writeDataToFile("Data/KodakDistanceLevelOne.pkl", distanceMatrix)
    util.writeDataToFile("Data/KodakLabelLevelOne.pkl", labels)









