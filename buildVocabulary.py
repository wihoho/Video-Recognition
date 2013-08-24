__author__ = 'GongLi'

# Read in all SIFT features
import os
import Utility as util
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle

stackOfAllFeatures = np.zeros(128).reshape(1, 128)
for label in os.listdir("/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak"):
    if label == ".DS_Store":
        continue

    path = "/Users/GongLi/Dropbox/FYP/Duan Lixin Data Set/sift_features/Kodak/" + label

    for video in os.listdir(path):
        if video == '.DS_Store':
            continue

        print label +": "+ video

        videoPath = path + "/" + video
        videoFeatures = util.readVideoData(videoPath, 100)
        # stackOfAllFeatures.append(videoFeatures)
        stackOfAllFeatures = np.vstack((stackOfAllFeatures, videoFeatures))

temp = stackOfAllFeatures[1:]
# Perform K-Means: 2500 centroids
kmeans = MiniBatchKMeans(init="k-means++", n_clusters=2500, n_init=10)
kmeans.fit(temp)
vocabulary = kmeans.cluster_centers_

# Save vocabulary
file = open("Data/voc.pkl", "w")
pickle.dump(vocabulary, file)

