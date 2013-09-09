__author__ = 'GongLi'

import os
import numpy as np
import Utility as util
from scipy.cluster.vq import *


def normalizeSIFT(descriptor):
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)

    if norm > 1.0:
        result = np.true_divide(descriptor, norm)
    else:
        result = None

    return result

def buildHistogramForVideo(pathToVideo, vocabulary):
    frames = os.listdir(pathToVideo)
    size = len(vocabulary)

    stackOfHistogram = np.zeros(size).reshape(1, size)
    for frame in frames:
        # build histogram for this frame
        completePath = pathToVideo +"/"+ frame
        lines = open(completePath, "r").readlines()

        print completePath

        histogram = np.zeros(size)
        for line in lines[1:]:
            data = line.split(" ")
            feature = data[4:]

            for i in range(len(feature)):
                item = int(feature[i])
                feature[i] = item

            feature = normalizeSIFT(feature)
            codes = nearestNeighbours(feature, vocabulary)

            for i in range(4):
                histogram[codes[i]] += 1.0 / (2**i)

        stackOfHistogram = np.vstack((stackOfHistogram, histogram.reshape(1,size)))

    return stackOfHistogram[1:]

def nearestNeighbours(feature, vocabulary):

    feature = feature.reshape(1,128)
    vocSize = vocabulary.shape[0]

    distances = []
    for i in range(vocSize):
        distances.append(np.linalg.norm(feature - vocabulary[i]))

    # return top 4 nearest indices
    codes = sorted(range(len(distances)), key=lambda i: distances[i])[:4]

    return codes


if __name__ == "__main__":
    vocabulary = util.loadObject("Data/voc.pkl")

    for label in os.listdir("/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"):
        if label == '.DS_Store':
            continue

        # if label in ["birthday", "parade", "picnic"]:
        #     continue

        path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak/" + label

        for video in os.listdir(path):
            # Create this file for later storage of histograms
            if video == ".DS_Store":
                continue

            file = open("KodakHistogramSoftWeight/"+label+"_"+video+".pkl", "w")

            videoPath = path +"/"+video
            print videoPath

            videoHistogram = buildHistogramForVideo(videoPath, vocabulary)

