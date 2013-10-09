__author__ = 'GongLi'


import os
import Utility as util
import time
import numpy as np
import subprocess as sub

from multiprocessing import Process

def C_EMD(feature1, feature2, excutablePath):

    H = feature1.shape[0]
    I = feature2.shape[0]

    distances = np.zeros((H, I))
    for i in range(H):
        for j in range(I):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])


    groundDistanceFile = open(excutablePath+"/groundDistance", "w")
    groundDistanceFile.write(str(H) +" "+ str(I) +"\n")

    distances = distances.reshape((H * I, 1))
    for i in range(H * I):
        groundDistanceFile.write(str(distances[i][0]) + "\n")

    groundDistanceFile.close()

    # Run C programme to calculate EMD
    # os.system("EarthMover")
    sub.call(excutablePath+"/EarthMover")

    # Read in EMD distance
    file = open(excutablePath+"/result", "r").readlines()

    groundDistanceFile.close()

    while True:
        try:
            os.remove(excutablePath+"/groundDistance")
            break

        except:
            time.sleep(1)
            print "groundDistance is not deleted properly!"


    return float(file[0])

def calculateDistanceAtLevelOne(videoOne, videoTwo, excutablePath):

    assert len(videoOne) == 8
    assert len(videoTwo) == 8

    distances = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            distances[i][j] = C_EMD(videoOne[i], videoTwo[j], excutablePath)

    unalignedDistance = 0.0
    for i in range(8):
        unalignedDistance += distances[i][i]
    unalignedDistance /= 8.0

    alignedDis = alignedDistance(distances, excutablePath)

    return alignedDis, unalignedDistance

def alignedDistance(distances, excutablePath):

    shape = distances.shape
    assert shape[0] == 8 and shape[1] == 8

    groundDistanceFile = open(excutablePath+"/groundDistance", "w")
    groundDistanceFile.write("8 8\n")
    distances = distances.reshape((64,1))
    for i in range(64):
        groundDistanceFile.write(str(distances[i][0]) + "\n")
    groundDistanceFile.close()

    sub.call(excutablePath +"/EarthMover")

    file = open(excutablePath+ "/result", "r").readlines()

    while True:
        try:
            os.remove(excutablePath+"/groundDistance")
            break
        except:
            time.sleep(1)
            print "groundDistance is not deleted properly!"

    return float(file[0])


def subCalculateDistances(videoListPath, processIndexes, outputFileName, EMD_executablePath):

    videosPath = util.loadObject(videoListPath)
    allVideoNum = len(videosPath)
    alignedDistanceFile = open("File/Aligned"+outputFileName, "w")
    unalignedDistanceFile = open("File/Unaligned"+outputFileName, "w")


    alignedKodakDis = util.loadObject("D:\GongLi (sce12-0548)\Projects\VideoRecognition-master\Data\LevelOne\KodakDistanceLevelOneAligned.pkl")
    unalignedKodakDis = util.loadObject("D:\GongLi (sce12-0548)\Projects\VideoRecognition-master\Data\LevelOne\KodakDistanceLevelOneUnAligned.pkl")


    for baseIndex in processIndexes:
        baseVideo = util.loadObject(videosPath[baseIndex])

        for candiateIndex in range(baseIndex + 1, allVideoNum, 1):

            if baseIndex < 195 and candiateIndex < 195:
                alignedDis = alignedKodakDis[baseIndex][candiateIndex]
                unalignedDis = unalignedKodakDis[baseIndex][candiateIndex]

            else:
                candiateVideo = util.loadObject(videosPath[candiateIndex])
                alignedDis, unalignedDis = calculateDistanceAtLevelOne(baseVideo, candiateVideo, EMD_executablePath)

            # Write into files
            alignedDistanceFile.write(str(baseIndex) +"\t"+ str(candiateIndex) +"\t"+ str(alignedDis) +"\n")
            unalignedDistanceFile.write(str(baseIndex) +"\t"+ str(candiateIndex) +"\t"+ str(unalignedDis) +"\n")
            print "[" +str(baseIndex) +","+ str(candiateIndex)+"]:" +"\t"+ str(alignedDis)

    print time.ctime()
    alignedDistanceFile.close()
    unalignedDistanceFile.close()


def calculateVideoDistanceLevelOne(videoListPath, numberOfProcess, EMDPaths):

    assert numberOfProcess == len(EMDPaths)

    videosPath = util.loadObject(videoListPath)
    videoSizes = len(videosPath)

    allIndexList = []
    for i in range(numberOfProcess):
        allIndexList.append([j for j in range(i, videoSizes, numberOfProcess)])

    processes = [Process(target=subCalculateDistances, args=(videoListPath, allIndexList[i], str(i)+"_process", EMDPaths[i])) for i in range(numberOfProcess)]
    for p in processes:
        p.start()

if __name__ == "__main__":

    videoListPath = "videoList.pkl"
    numberOfProcess = 4
    EMD_Paths = ["EMDone", "EMDtwo", "EMDthree", "EMDfour"]

    calculateVideoDistanceLevelOne(videoListPath, numberOfProcess, EMD_Paths)











