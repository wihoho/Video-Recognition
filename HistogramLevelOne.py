__author__ = 'GongLi'

import Utility as util
import numpy as np
from scipy.cluster.vq import *
import os
import time

def buildHistogramLevelOne(videoPath):

    frames = os.listdir(videoPath)
    boundary = len(frames) / 2

    voc = util.loadObject("Data/voc.pkl")
    vocSize = len(voc)

    front = [np.zeros(vocSize).reshape(1, vocSize)] * 4
    back = [np.zeros(vocSize).reshape(1, vocSize)] * 4

    # process front
    for frame in frames[:boundary]:
        framePath = videoPath +"/"+ frame
        frameHistograms = imageHistogramLevelOne(framePath, voc)
        if not frameHistograms:
            continue

        assert len(frameHistograms) == 4

        for i in range(4):
            front[i] = np.vstack((front[i], frameHistograms[i]))

    for i in range(len(front)):
        front[i] = front[i][1:]

    # process back
    for frame in frames[boundary: ]:
        framePath = videoPath +"/"+ frame
        frameHistograms = imageHistogramLevelOne(framePath, voc)
        if not frameHistograms:
            continue

        for i in range(4):
            back[i] = np.vstack((back[i], frameHistograms[i]))

    for i in range(len(back)):
        back[i] = back[i][1:]

    VideoEightHistograms = front + back
    return VideoEightHistograms

def imageHistogramLevelOne(framePath, voc):

    completePath = framePath
    lines = open(completePath, "r").readlines()

    frameFeatures = np.zeros(128).reshape(1, 128)
    featureLocations = np.zeros(2).reshape(1, 2)
    for line in lines[1:]:
        data = line.split(" ")
        feature = data[4:]
        locations = data[:2]

        for i in range(len(feature)):
            item = int(feature[i])
            feature[i] = item

        for i in range(len(locations)):
            item = float(locations[i])
            locations[i] = item

        feature = util.normalizeSIFT(feature)
        frameFeatures = np.vstack((frameFeatures, feature))
        featureLocations = np.vstack((featureLocations, locations))

    frameFeatures = frameFeatures[1:]
    if len(frameFeatures) == 0:
        return None

    featureLocations = featureLocations[1:]

    maxValues = np.amax(featureLocations, axis=0)
    rowMax = maxValues[0]
    columnMax = maxValues[1]

    if rowMax > 240 or columnMax > 320:
        width = 640
        height = 480

    else:
        width = 320
        height = 240

    # construct the histograms
    vocSize =  len(voc)

    list1 = []
    list2 = []
    list3 = []
    list4 = []

    halfWidth = width / 2.0
    halfHeight = height / 2.0

    for i in range(featureLocations.shape[0]):
        position = featureLocations[i]
        x = position[1]
        y = position[0]

        if x < halfWidth and y < halfHeight:
            list1.append(i)
        elif x >= halfWidth and y < halfHeight:
            list2.append(i)
        elif x < halfWidth and y >= halfHeight:
            list3.append(i)
        else:
            list4.append(i)

    featureSectionOne = frameFeatures[list1]
    featureSectionTwo = frameFeatures[list2]
    featureSectionThree = frameFeatures[list3]
    featureSectionFour = frameFeatures[list4]

    allFourHistogram = []
    histogram = np.zeros(vocSize)
    codes, distance = vq(featureSectionOne, voc)
    for code in codes:
        histogram[code] += 1
    allFourHistogram.append(histogram)

    histogram = np.zeros(vocSize)
    codes, distance = vq(featureSectionTwo, voc)
    for code in codes:
        histogram[code] += 1
    allFourHistogram.append(histogram)

    histogram = np.zeros(vocSize)
    codes, distance = vq(featureSectionThree, voc)
    for code in codes:
        histogram[code] += 1
    allFourHistogram.append(histogram)

    histogram = np.zeros(vocSize)
    codes, distance = vq(featureSectionFour, voc)
    for code in codes:
        histogram[code] += 1
    allFourHistogram.append(histogram)

    return allFourHistogram

if __name__ == "__main__":

    for label in os.listdir("/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"):
        path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak/" + label
        print path

        if label in ["birthday", "parade","picnic", ".DS_Store"]:
            continue


        for video in os.listdir(path):
            if video == ".DS_Store":
                continue

            if label == "show" and video in ["100_0204", "100_0205", "100_0207", "100_0208"]:
                continue

            fileName = "KodakLevelOneHistograms/"+label+"_"+video+".pkl"

            videoPath = path +"/"+video
            print videoPath

            eightHistograms = buildHistogramLevelOne(videoPath)
            util.storeObject(fileName, eightHistograms)

        print time.ctime()