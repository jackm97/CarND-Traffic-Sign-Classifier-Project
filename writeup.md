# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[bar]: ./writeup_images/bar_graph.JPG 
[examples]: ./writeup_images/example_images.JPG
[preprocessed]: ./writeup_images/preprocessing.JPG
[preprocessed_test]: ./writeup_images/preprocessing_test.JPG
[image1]: ./test_signs/test_image1.jpg "Traffic Sign 1"
[image2]: ./test_signs/test_image2.jpg "Traffic Sign 2"
[image3]: ./test_signs/test_image3.jpg "Traffic Sign 3"
[image4]: ./test_signs/test_image4.jpg "Traffic Sign 4"
[image5]: ./test_signs/test_image5.jpg "Traffic Sign 5"  

---
### Data Set Summary & Exploration

#### 1. Basic Summary of Dataset

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization of Dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread across the training and validation datasets.

![alt text][bar]

Here are a few example images from the traffic sign dataset.

![alt_text][examples]

### Design and Test a Model Architecture

#### 1. Preprocessing

For the preprocessing step I resized the images from 32x32 to 48x48. I then used [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE) with a clip limit of 2.0 and a 6x6 grid to increase image contrast which has been shown to improve training [(Cireşan, Meier, Masci & Schmidhuber, 2011)](https://ieeexplore.ieee.org/abstract/document/6033458). Then the images were globally normalized by mean subtraction and standard deviation division. Here are the previous images after preprocessing:

![alt_text][preprocessed]

#### 2. Model Architecture

My initial architecture outline was derived from [(Cireşan, Meier, Masci & Schmidhuber, 2011)](https://ieeexplore.ieee.org/abstract/document/6033458). I found that changing the activation functions to RELUs, moving the max-pooling layers before the activation layers and adding a local response normalization(LRN) in between the pooling and activation layers decreased training time and increased generalization. Additionally a dropout layer(`keep_prob=.7`) was used instead of LRN since dropout better increased generalization without decreasing accuracy on the training set.

| Layer         		|     # of Maps & Neurons	        					| Kernel   |
|:---------------------:|:-------------------------------:|:--------------:| 
| 0         		| 3 maps of 48x48 neurons  							|                |
| Convolutional     	| 100 maps of 46x46 neurons 	|    3x3         |
| Max pooling	      	| 100 maps of 23x23 neurons 				|   2x2        |
| LRN	    | 100 maps of 23x23 neurons      									|    |
| RELU					|	100 maps of 23x23 neurons											|               |
| Convolutional     	| 150 maps of 20x20 neurons 	|    4x4        |
| Max pooling	      	| 150 maps of 10x10 neurons 				|   2x2        |
| LRN	    | 150 maps of 10x10 neurons      								|   |
| RELU					|	150 maps of 10x10 neurons											|               |
| Convolutional     	| 250 maps of 8x8 neurons 	|    3x3         |
| Max pooling	      	| 250 maps of 4x4 neurons 				|   2x2        |
| LRN	    | 250 maps of 4x4 neurons      									|    |
| RELU					|	250 maps of 4x4 neurons											|               |
| Flatten     	| 4000 neurons 	|             |
| Fully Connected     	| 200 neurons 	|             |
| RELU					|	200 neurons											|               |
| Dropout	    | 200 neurons     									|
| Fully Connected     	| 43 logits 	|             |
 


#### 3. Training

The model was trained on an NVIDIA Tesla T4 using Google's cloud computing service. To train the model, I used Adam optimization with a learning rate of .004 to minimize the softmax cross entropy loss calculated from the logits from the previous step. A batch size of 50 produced the best results when training for 10 epochs. To improve accuracy and generalization, the input images were individually translated and rotated at random from a range of +/- 10% and +/- 10 degrees respectively [(Cireşan, Meier, Masci & Schmidhuber, 2011)](https://ieeexplore.ieee.org/abstract/document/6033458).

#### 4. Approach and Results on Train, Validation and Test Sets

My final model results were:
* training set accuracy of .999
* validation set accuracy of .989 
* test set accuracy of .977

As mentioned above, the architecture used was derived from [(Cireşan, Meier, Masci & Schmidhuber, 2011)](https://ieeexplore.ieee.org/abstract/document/6033458). This article is written by on of the top teams for this particular dataset's classification competition. The results of this competition, among other information, can be found [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results).

The final model's accuracy on the training and validation set consistently stay within a few percent of eachother, and are both increasing throughout training indicating my model is not overfitting. The test set accuracy is lower by about 1% so the model will likely work well on new images.
 

### Test a Model on New Images

#### 1. Five German Signs From Google Images

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

They should all be relatively simple to classify except for the images with watermarks. These may end up getting classified incorrectly since the training set didn't have watermarks. Additionally, these are higher resolution images with different aspect ratios so downscaling them to 48x48 images may cause the classifier to incorrectly classify the signs.

Here are examples of the preprocessed test images:

![alt text][preprocessed_test]

#### 2. Results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| 30 km/hr   			| Stop 										|
| Stop					| Stop										|
| Road work      		| Road work					 				|
| Yield			| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The test set had an accuracy of 97.7%, meaning this sample had lower accuracy. However, the sample size is much smaller; a larger sample would likely bring the accuracy to a similar value as the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The first image is classified with near 100% confidence and is classified correctly:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00         			| Keep right   									| 
| 5.44e-7     				| Keep left 										|
| 1.15e-10					| Yield											|
| 4.39e-10      			| Roundabout mandatory					 				|
| 1.34e-10				    | Turn left ahead      							|

The second image is classified incorrectly with 43.4% certainty likely due to the aspect ratio of the original image not being square causing a scaling issue when resizing to 48x48:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .434        			| Stop  									| 
| .357    				| 30 km/hr 										|
| .116				| 80 km/hr										|
| .040      			| End of speed limit (80km/hr)					 				|
| .035			    | Yield      							|

The third image of a stop sign is classified correctly as a stop sign with nearly 100% confidence:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00         			| Stop 									| 
| 1.44e-7    				| No passing										|
| 9.14e-9				| Dangerous curve to the right											|
| 8.78e-9      			| 60 km/hr				 				|
| 4.46e-9			    | Yield      							|

The fourth image was classified correctly as road work sign with nearly 100% confidence:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00        			| Road work  									| 
| 7.09e-5    				| Dangerous curve to the right										|
| 2.41e-5				| Stop											|
| 1.66e-5	      			| Bumpy road					 				|
| 1.76e-6			    | Traffic signals     							|

The last image was classified correctly with almost 100% confidence:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00        			| Yield  									| 
| 1.73e-4    				| 100 km/hr									|
| 4.74e-5				| 80 km/hr											|
| 1.25e-5      			| Stop					 				|
| 2.36e-6			    | 50 km/hr      							|


