__author__ = 'GongLi'

import os
import numpy as np
from scipy.cluster.vq import *
import pickle

def normalizeSIFT(descriptor):
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)

    if norm > 1.0:
        result = np.true_divide(descriptor, norm)

    return result

# Read in video frames under a folder
def readVideoData(pathOfSingleVideo, subSampling = 5):
    frames = os.listdir(pathOfSingleVideo)

    stackOfSIFTFeatures = []
    for frame in frames:
        completePath = pathOfSingleVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        for line in lines[1::subSampling]:
            data = line.split(" ")
            feature = data[4:]
            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            # normalize SIFT feature
            feature = normalizeSIFT(feature)
            stackOfSIFTFeatures.append(feature)

    return np.array(stackOfSIFTFeatures)

def buildHistogramForVideo(pathToVideo, vocabulary):
    frames = os.listdir(pathToVideo)
    size = len(vocabulary)

    stackOfHistogram = np.zeros(size).reshape(1, size)
    for frame in frames:
        # build histogram for this frame
        completePath = pathToVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        frameFeatures = np.zeros(128).reshape(1, 128)
        for line in lines[1:]:
            data = line.split(" ")
            feature = data[4:]

            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            feature = normalizeSIFT(feature)
            frameFeatures = np.vstack((frameFeatures, feature))

        frameFeatures = frameFeatures[1:]
        codes, distance = vq(frameFeatures, vocabulary)

        histogram = np.zeros(size)
        for code in codes:
            histogram[code] += 1

        stackOfHistogram = np.vstack((stackOfHistogram, histogram.reshape(1,size)))

    return stackOfHistogram[1:]

def writeDataToFile(filePath, data):
    file = open(filePath, "w")
    pickle.dump(data, file)
    file.close()

def loadDataFromFile(filePath):
    file = open(filePath, 'r')
    data = pickle.load(file)
    return data


if __name__ == "__main__":
    import pickle
    file = open("Data/birthday_103_0337.pkl", "r")

    data = pickle.load(file)
    print "Yes"










