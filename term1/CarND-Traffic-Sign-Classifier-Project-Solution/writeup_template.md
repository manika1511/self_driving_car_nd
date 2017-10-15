# **Traffic Sign Recognition**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is ?
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 as it is rgb scale
* The number of unique classes/labels in the data set is 43

I did an exploratory visualization of the dataset to see the distribution of different classes in the dataset. 
The two bar chart showing the distribution of different classes in the train dataset. It can be seen that some classes like classes 1-6, 25 and 38 have more number of instances as compared to other classes. Similar distribution can be seen in the test dataset. So, test data set is a good representation of train dataset.

![train_dist](https://user-images.githubusercontent.com/20146538/31580212-d74714f8-b0fd-11e7-91f2-a3b646d7fa0a.png)

![test_dist](https://user-images.githubusercontent.com/20146538/31580239-b7ee1682-b0fe-11e7-8b46-b3de4cdaa409.png)

Also, to have a general idea about the type of images in the dataset, I displayed 12 random images from the dataset which are shown below.

<img width="913" alt="screen shot 2017-10-14 at 4 26 18 pm" src="https://user-images.githubusercontent.com/20146538/31580265-4c1e4e3a-b0ff-11e7-8d2a-a8aafcde800e.png">

### Design and Test a Model Architecture

#### 1. Data Pre-processing

As a first step, I decided to convert the images to grayscale as working with grayscale images is easier and computations are faster. 
Here is an example of a traffic sign image before and after grayscaling.
<img width="225" alt="r" src="https://user-images.githubusercontent.com/20146538/31584515-bc993606-b164-11e7-939b-235b70515d15.png">
<img width="193" alt="g" src="https://user-images.githubusercontent.com/20146538/31584516-bdd7ed82-b164-11e7-8730-0fc7ab634f09.png">

As a last step, I normalized the image data to bring all the values between -1 and 1 so that the neural network is trained properly with the set learning rate. Attributes with different scales posses difficulty in the network learning.

I decided to generate additional data because the distribution of dataset has some classes with way more samples as compared to others. Also, generating more data with some variation in axis gives more varied samples. 

To add more data to the the data set, I used the random rotation because the signs with different angles will help in better training of the model.
Here is an example of an original image and an augmented image:

![rotation1](https://user-images.githubusercontent.com/20146538/31584583-01f29f98-b166-11e7-836e-726d17051ace.png)

![ro](https://user-images.githubusercontent.com/20146538/31584584-032b2312-b166-11e7-808e-d83760700c26.png)

The difference between the original data set and the augmented data set is that the augmented dataset has more samples for the ones which had very less samples which helps in training the model with better accuracy.


#### 2. Final model Architecture
My final model architecture is dapted from Sermanet/LeCunn traffic sign classification journal article http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf.

<img width="332" alt="screen shot 2017-10-15 at 5 09 00 am" src="https://user-images.githubusercontent.com/20146538/31584632-0fc5624e-b167-11e7-92ab-eac5365e3500.png">

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For training the model, I used started with a batch size of 150 but after optimization I landed up with 100 which gave me a better result. Number of Epochs 

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


