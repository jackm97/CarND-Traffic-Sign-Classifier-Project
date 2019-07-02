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

[bar]: ./writeup_images/visualization.jpg "Visualization"
[examples]: ./writeup_images/grayscale.jpg "Examples"
[global_norm]: ./writeup_images/random_noise.jpg "Global Norm"
[local_norm]: ./writeup_images/placeholder.png "Local Norm"
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

In the first step I performed a global normalization on each image using `scikit-learn.preprocessing.normalize()`:

![alt_text][global_norm]

Next I converted the images to grayscale and then performed a local normalization using `skimage.filiters.rank.equalize()`:

![alt_text][local_norm]

The motivation behind the normalization steps are described in [(Garcia, Garcia & Soria-Morillo, 2018)](https://www.sciencedirect.com/science/article/pii/S0893608018300054).

Finally, I use `cv2.resize()` to resize each image to (48,48,3) for the CNN architecture described in the next step.


#### 2. Model Architecture

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

 


#### 3. Training

To train the model, I used Adam optimization with a learning rate of .004 to minimize the softmax cross entropy loss calculated from the logits from the previous step. A batch size of 200 produced the best results when training for 10 epochs from the three batch sizes I tried (50,200,1000). After each activation (including max pooling and normalization), dropout with a keep probability of .5 was used during training

#### 4. Approach and Results on Train, Validation and Test Sets

My final model results were:
* training set accuracy of .997
* validation set accuracy of .983 
* test set accuracy of .962

As mentioned above, the architecture used was a combination of [(Garcia, Garcia & Soria-Morillo, 2018)](https://www.sciencedirect.com/science/article/pii/S0893608018300054) and [(Cireşan, Meier, Masci & Schmidhuber, 2011)](https://ieeexplore.ieee.org/abstract/document/6033458), where the former mentions adding a normalization after each activation layer and the latter describes the rest of the layers. These two articles are written by the winning teams for this particular dataset's classification competition. The results of this competition, among other information, can be found [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results).

The final model's accuracy on the training and validation set consistently stay within a few percent of eachother, and are both increasing throughout training indicating my model is not overfitting. The test set accuracy is lower but still well above 93% so the model will likely work well on new images.
 

### Test a Model on New Images

#### 1. Five German Signs From Google Images

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

They should all be relatively simple to classify except for the imaes with watermarks. These may end up getting classified incorrectly since the training set didn't have watermarks. Additionally, these are higher resolution images of traffic signs, and downscaling them to 48x48 images may cause the classifier to incorrectly classify the signs.

#### 2. Results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right   									| 
| 30 km/hr   			| 30 km/hr 										|
| Stop					| Priority Road										|
| Road work      		| Road work					 				|
| Yield			| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The test set had an accuracy of 96%, meaning this sample had lower accuracy. However, the sample size is much smaller; a larger sample would likely bring the accuracy to a similar value as the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The first image is classified with near 100% certainty and is classified correctly:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right   									| 
| 1.24e-10     				| Turn left ahead 										|
| 5.63e-13					| Go straight or right											|
| 2.44e-13	      			| Priority road					 				|
| 1.30e-13				    | Yield      							|

The second image is classified correctly with 33% certainty likely due to the aspect ratio of the original image not being square causing a scaling issue when resizing to 48x48:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .33         			| 30 km/hr   									| 
| .23    				| 60 km/hr 										|
| .11				| 80 km/hr											|
| .08	      			| General caution					 				|
| .07			    | Yield      							|

The third image of a stop sign is classified incorrectly as a priority road sign with a certainty of 99.7%. This incorrect classification is interesting because of the high confidence value and should be explored more deeply to possibly improve model performance for other similar cases. Data augmentation/more data and better preprocessing would likely be the most influential in producing better results for cases like this.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| Priority road  									| 
| .001    				| Turn left ahead										|
| 5.2e-4				| Roundabout mandatory											|
| 4.0e-4	      			| Yield					 				|
| 3.7e-4			    | Keep right      							|

The fourth image was classified correctly as road work sign with 93% confidence:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .93         			| Road work  									| 
| .03    				| Traffic signals										|
| .03				| Bumpy road											|
| 2.2e-3	      			| Beware of ice/snow					 				|
| 5.1e-4			    | Bicycles crossing     							|

The last image was classified correctly with almost 100% confidence:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9997         			| Yield  									| 
| 1.4e-5    				| 50 km/hr									|
| 5.4e-6				| 80 km/hr											|
| 3.7e-6      			| Priority road					 				|
| 1.7e-6			    | Stop      							|


