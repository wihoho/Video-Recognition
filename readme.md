This repository contains the source code of my final year project about video recognition. Video recognition aims to recognise various videos based on their content (or frames). If we treat video as a sequence of frames, we then could use several frames to represent a video. At the same time, each image can be converted into a histogram using Bag of Words framework. Finally, a stack of histograms is used to represent a video. Based on this kind of representation, Earth Mover's Distance could be employed to measure video-to-video distance. 

### Dependencies

* python 2.7.x
* numpy
* scipy
* sklearn
* ....

### Data Set

[SIFT](http://vc.sce.ntu.edu.sg/index_files/VisualEventRecognition/features.html) 

### Simple Documentation

To have a better understanding about this project, please check 

* [Report](https://dl.dropboxusercontent.com/u/37572555/Github/FYP%20Report/FinalReportV4.pdf)
* [Slides](https://dl.dropboxusercontent.com/u/37572555/Github/FYP%20Report/SlidesV2.pdf)

### Function of each python module

#### 1. Build visual vocabulary
* `buildVocabulary.py`: build visual vocabulary       


#### 2. Build histograms
* `buildHistogram.py`: histogram built in Naive Bag of Words

* `BuildHistogramForDifferentCriteria.py`: histogram built with various weighting schemes

* `HistogramSoftWeight.py`: histograms using soft assignment

* `HistogramLevelOne.py`: histograms built for Aligned Space-Time Pyramid Matching at level 1

#### 3. Measure video-to-video distance
* `calculateDistanceMatrix.py`: employ **Earth Mover's Distance** to calculate video-video distance.

* `calculateDistanceMatrixLevelOne.py`: Aligned Space-Time Pyramid Matching at level 1

* Note on implementation of **Earth Mover's Distance**
	* Adapt the author's [C implementation](http://www.cs.duke.edu/~tomasi/software/emd.htm) with python module through a file interface
	* The modified C codes are under the folder `EarthMoverDistance SourceCode`

#### 4. Classify based on SVM

* `testDriver.py`: employ SVM and K Fold for experiments