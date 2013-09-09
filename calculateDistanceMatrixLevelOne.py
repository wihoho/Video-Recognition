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

    unaligedDistance = 0.0
    for i in range(8):
        unaligedDistance += distances[i][i]
    unaligedDistance = unaligedDistance / 8.0

    # calculate aligned distance
    alignedDis = alignedDistances(distances)

    return alignedDis, unaligedDistance

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
    sub.call("/Users/GongLi/PycharmProjects/VideoRecognition/EarthMoverDistance")

    # Read in EMD distance
    file = open("result", "r").readlines()
    os.remove("groundDistance")

    return float(file[0])

def calculateDistanceMatrixAtLevelOne(PATH):
    alignedDistanceFile = open("Data/all_alignedDistanceFile", "w")
    unalignedDistanceFile = open("Data/all_unalignedDistanceFile", "w")

    labels = []
    videoData = []
    identifiers = []

    for category in os.listdir(PATH):

        if category == ".DS_Store":
            continue

        tempPath = PATH +"/"+category

        for domain in os.listdir(tempPath):
            if domain == ".DS_Store":
                continue

            P = tempPath +"/"+ domain
            print P

            for video in os.listdir(P):
                if video == ".DS_Store":
                    continue

                completePath = P +"/"+ video
                print completePath

                videoHistograms = util.loadObject(completePath)
                videoData.append(videoHistograms)

                labels.append(domain)
                identifiers.append(category)

    numberOfVideos = len(labels)
    alignedDistanceMatrix = np.zeros((numberOfVideos, numberOfVideos))
    unalignedDistanceMatrix = np.zeros((numberOfVideos, numberOfVideos))

    for i in range(numberOfVideos):
        for j in range(i, numberOfVideos, 1):

            if i == j:
                unalignedDistanceMatrix[i][j] = 0
                alignedDistanceMatrix[i][j] = 0

                alignedDistanceFile.write(str(0) +"\t")
                unalignedDistanceFile.write(str(0) +"\t")
                continue

            aligned, unaligned = calculateDistanceAtLevelOne(videoData[i], videoData[j])

            alignedDistanceMatrix[i][j] = aligned
            alignedDistanceMatrix[j][i] = alignedDistanceMatrix[i][j]

            unalignedDistanceMatrix[i][j] = unaligned
            unalignedDistanceMatrix[j][i] = unalignedDistanceMatrix[i][j]

            print "unaligned :["+str(i) +","+ str(j)+"]: " + str(unalignedDistanceMatrix[i][j])
            print "aligned :["+str(i) +","+ str(j)+"]: " + str(alignedDistanceMatrix[i][j])

            alignedDistanceFile.write(str(alignedDistanceMatrix[i][j]) +"\t")
            unalignedDistanceFile.write(str(unalignedDistanceMatrix[i][j]) +"\t")

        alignedDistanceFile.write("\n")
        unalignedDistanceFile.write("\n")


    return alignedDistanceMatrix, unalignedDistanceMatrix, labels

if __name__ == "__main__":

    alignedDistanceMatrix, unalignedDistance, labels = calculateDistanceMatrixAtLevelOne("/Users/GongLi/PycharmProjects/VideoRecognition/VideoHistogramLevel1")
    util.storeObject("Data/ALL_DistanceLevelOneAligned.pkl", alignedDistanceMatrix)
    util.storeObject("Data/ALL_DistanceLevelOneUnAligned.pkl", unalignedDistance)

    util.storeObject("Data/ALL_LabelLevelOne.pkl", labels)









