__author__ = 'GongLi'

import os
import Utility as util
from sklearn import preprocessing
import numpy as np

if __name__ == "__main__":

    path = "HistogramDifferentWeightingSchemes/txx/"
    N = 0
    n = np.zeros((1, 2500))

    for label in os.listdir(path):
        labelPath = path + label

        for video in os.listdir(labelPath):

            videoPath = labelPath +"/"+ video
            videoHistogram = util.loadObject(videoPath)


            # bxx
            binarizer = preprocessing.Binarizer()
            binaryHistogram = binarizer.transform(videoHistogram)
            binaryPath = "HistogramDifferentWeightingSchemes/bxx/" +label+ "/" +video
            util.storeObject(binaryPath, binaryHistogram)

            # txc
            tfN = np.copy(videoHistogram)
            row, column = tfN.shape

            temSums = np.sum(tfN, axis=1)
            for i in range(row):
                for j in range(column):
                    tfN[i][j] = float(tfN[i][j]) / temSums[i]
                    print str(tfN[i][j])
            
            tfNPath = "HistogramDifferentWeightingSchemes/txc/" +label+ "/" +video
            util.storeObject(tfNPath, tfN)

            # Update parameters
            N += videoHistogram.shape[0]

            binarySums = np.sum(binaryHistogram, axis=0)
            n = np.vstack((n, binarySums))

    n = n[1:]
    n = np.sum(n, axis=0)

    tfxLogParameters = np.zeros((1, n.shape[1]))
    for i in range(n.shape[1]):
        if n[i] != 0:
            tfxLogParameters[0][i] = np.log(N / n[i])

    # tfx
    for label in os.listdir(path):
        labelPath = path + label

        for video in os.listdir(labelPath):

            videoPath = labelPath +"/"+ video
            videoHistogram = util.loadObject(videoPath)

            # tfx
            tfx = np.copy(videoHistogram)
            for i in range(tfx.shape[0]):
                tfx[i] = tfx[i] * tfxLogParameters

            tfxPath = "HistogramDifferentWeightingSchemes/tfx/" +label+ "/" +video
            util.storeObject(tfxPath, tfx)

            # tfc
            temSum = np.sum(tfx, axis=1)
            row, column = tfx.shape
            for i in range(row):
                for j in range(column):
                    tfx[i][j] = float(tfx[i][j]) / temSum[i]

            tfcPath = "HistogramDifferentWeightingSchemes/tfc/" +label+ "/" +video
            util.storeObject(tfcPath, tfx)



