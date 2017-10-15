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
* The size of the validation set is 4410
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
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution1 5x5     	| 1x1 stride, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution2 5x5     	| 1x1 stride, outputs 10x10x16 	|
| RELU 				|												|
| Max pooling	(A)      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution3 5x5     	| 1x1 stride, outputs 1x1x400 	|
| RELU		(B)			|												|
| Flatten	output from A and B				|	output both 400											|
| Concatenate flattened layers		|     output 800   									|
|	Dropout Layer			|												|
|		Fully connected layer				|			output 43									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For training the model, I used started with a batch size of 170 but after optimization I landed up with 128 which gave me a better result. Number of Epochs, started of with 30 due to machine limitation. Then, used nvidia CUDA and tired 80 Epochs and got accuracy above 94%.

To train the model, I started with a learning rate of 0.0008 which gave me very slow output. Then, I increased it to 0.001 and got the desired performance. 

I used AdamOptimizer for error minimization. I also tried GradientDescentOptimizer, but it didn't give good results.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

First I chose the simple LeNet architecture studied in the class. I ran it for 50 epochs, but it gave around 88% accuracy. I tried adding dropout layer of 0.7 but it didn't show any significant improvement. I even tried adjusting the batch size and learning rate. But, accuracy didn't cross 90%. I was unable to access AWS instance so it took a lot of time to train. So, I searched on net and studied different architectures which have been used for such problems. It was then, I came across "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Pierre Sermanet and Yann LeCun where they had suggested an architecture for traffic signal classification. 

I implemented the suggested model and it showed significant improvement. In first 10 epochs, the accuracy was already above 93%. In this architecture three convolution layers were used with max pooling. Layer flatenning and layer concatenation followed by a dropout and fully connected layer. 

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


