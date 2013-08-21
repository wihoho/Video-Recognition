__author__ = 'GongLi'

import os
import pickle
import Utility as util


file = open("Data/voc.pkl", "r")
vocabulary = pickle.load(file)

for label in os.listdir("/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"):
    if label == '.DS_Store':
        continue

    if label in ["birthday", "parade", "picnic"]:
        continue

    path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak/" + label

    for video in os.listdir(path):
        # Create this file for later storage of histograms
        if video == ".DS_Store":
            continue

        file = open("Data/"+label+"_"+video+".pkl", "w")

        videoPath = path +"/"+video
        print videoPath

        videoHistogram = util.buildHistogramForVideo(videoPath, vocabulary)
        pickle.dump(videoHistogram, file)
        file.close()



