# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[bar]: ./examples/visualization.jpg "Visualization"
[examples]: ./examples/grayscale.jpg "Grayscaling"
[global_norm]: ./examples/random_noise.jpg "Random Noise"
[local_norm]: ./examples/placeholder.png "Local Norm"
[image1]: ./test_signs/test_image1.jpg "Traffic Sign 1"
[image2]: ./test_signs/test_image2.jpg "Traffic Sign 2"
[image3]: ./test_signs/test_image3.jpg "Traffic Sign 3"
[image4]: ./test_signs/test_image4.jpg "Traffic Sign 4"
[image5]: ./test_signs/test_image5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread across the training and validation datasets.

![alt text][bar]

Here are a few example images from the traffic sign dataset.

![alt_text][examples]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In the first step I performed a global normalization on each image using `scikit-learn.preprocessing.normalize()`:

![alt_text][global_norm]

Next I performed a local normalization using `skimage.filiters.rank.equalize()`:

![alt_text][local_norm]

The motivation behind the normalization steps are described in [(Garcia, Garcia & Soria-Morillo, 2018)](https://www.sciencedirect.com/science/article/pii/S0893608018300054).

Finally, I use `cv2.resize()` to resize each image to (48,48,3) for the CNN architecture described in the next step.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers derived from [(Garcia, Garcia & Soria-Morillo, 2018)](https://www.sciencedirect.com/science/article/pii/S0893608018300054) and [(Cireşan, Meier, Masci & Schmidhuber, 2011)](https://ieeexplore.ieee.org/abstract/document/6033458):

| Layer         		|     # of Maps & Neurons	        					| Kernel   |
|:---------------------:|:-------------------------------:|:--------------:| 
| 0         		| 3 maps of 48x48 neurons  							|                |
| Convolutional     	| 100 maps of 46x46 neurons 	|    3x3         |
| RELU					|	100 maps of 46x46 neurons											|               |
| Max pooling	      	| 100 maps of 23x23 neurons 				|   2x2        |
| Local norm	    | 100 maps of 23x23 neurons      									|
| Convolutional     	| 150 maps of 20x20 neurons 	|    4x4        |
| RELU					|	150 maps of 20x20 neurons											|               |
| Max pooling	      	| 150 maps of 10x10 neurons 				|   2x2        |
| Local norm	    | 150 maps of 10x10 neurons      									|
| Convolutional     	| 250 maps of 8x8 neurons 	|    3x3         |
| RELU					|	250 maps of 8x8 neurons											|               |
| Max pooling	      	| 250 maps of 4x4 neurons 				|   2x2        |
| Local norm	    | 100 maps of 23x23 neurons      									|
| Fully Connected     	| 200 neurons 	|             |
| RELU					|	200 neurons											|               |
| Fully Connected     	| 43 logits 	|             |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimization with a learning rate of .004 to minimize the softmax cross entropy loss calculated from the logits from the previous step. A batch size of 200 produced the best results when training for 10 epochs from the three batch sizes I tried (50,200,1000). After each activation (including max pooling and normalization), dropout with a keep probability of .5 was used during training

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of .997
* validation set accuracy of .983 
* test set accuracy of .962

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

As mentioned above, the architecture used was a combination of [(Garcia, Garcia & Soria-Morillo, 2018)](https://www.sciencedirect.com/science/article/pii/S0893608018300054) and [(Cireşan, Meier, Masci & Schmidhuber, 2011)](https://ieeexplore.ieee.org/abstract/document/6033458), where the former mentions adding a normalization after each activation layer and the latter describes the rest of the layers. These two articles are written by the winning teams for this particular dataset's classification competition. The results of this competition, among other information, can be found [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results).

The final model's accuracy on the training and validation set consistently stay within a few percent of eachother, and are both increasing throughout training indicating my model is not overfitting. The test set accuracy is lower but still well above 93% so the model will likely work well on new images.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image4]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


