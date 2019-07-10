## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I use what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I then try out your model on images of German traffic signs that I found on the web.

I have included an Ipython notebook that contains further details, code and training results [here](./Traffic_Sign_Classifier.ipynb). 

I have also included a writeup detailing my thought process behind the preprocessing and network architecture along with a discussion of the results in a [writeup](./writeup.md). 

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
To run the IPython notebook:

* [OpenCV 4.0+](https://opencv.org/opencv-4-0/)
* [Tensorflow 1.14+](https://www.tensorflow.org/install) (NOTE: Some functions are deprecated and minor changes may need to be made for higher versions)
* [NumPy](https://www.numpy.org/)
* [sklearn .21+](https://scikit-learn.org/stable/)

Earlier versions of each of these except Tensorflow will likely work as well. This is simply how my machine was setup.


The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the [data set](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and separate the data into three pickle files `train.p`, `valid.p`, and `test.p` in the main directory. Since the train, validation, and tests sets won't be the same as the ones I used, expect slightly different results when running the IPython notebook.

