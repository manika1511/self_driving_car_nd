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

It can also be seen that some classes have so many samples (above 1000) and others have minimum of 180. So, data augmentation is required so that the classes with less samples may also have good representation.

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

To add more data to the the data set, I used the random rotation because the signs with different angles will help in better training of the model and generated images for classes with less than 300 samples.
Here is an example of an original image and an augmented image:

![rotation1](https://user-images.githubusercontent.com/20146538/31584583-01f29f98-b166-11e7-836e-726d17051ace.png)

![ro](https://user-images.githubusercontent.com/20146538/31584584-032b2312-b166-11e7-808e-d83760700c26.png)

The difference between the original data set and the augmented data set is that the augmented dataset has more samples for the ones which had very less samples which helps in training the model with better accuracy.

![download](https://user-images.githubusercontent.com/20146538/31589518-c259fc44-b1b7-11e7-93fe-c534ada6139a.png)


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

#### 4. Approach

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.955 
* test set accuracy of 0.875

First I chose the simple LeNet architecture studied in the class. I ran it for 30 epochs, 150 batch size, 0.001 learning rate but it gave around 93% accuracy. The accuracy was satisfactory but the initial accuracy was low. So, I searched on net and studied different architectures which have been used for such problems. It was then, I came across "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Pierre Sermanet and Yann LeCun where they had suggested an architecture for traffic signal classification. 

I implemented the suggested model and it showed significant improvement. In first 10 epochs, the accuracy was already above 93%. In this architecture three convolution layers were used with max pooling. Layer flatenning and layer concatenation followed by a dropout and fully connected layer. I experimented with different learning rate, batch size and epochs. I got the best result with learning rate 0.001, batch size 180 and epochs 30.

I learned a lot about Nvidia CUDA and cudnn with tensorflow-gpu. This helped me in running my training model faster and cheaper way with so many iterations.  

###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![30_kmp](https://user-images.githubusercontent.com/20146538/31588409-47dc5b62-b1a6-11e7-8ef0-04aee519ce33.png)
![left_turn](https://user-images.githubusercontent.com/20146538/31588392-088401a4-b1a6-11e7-9194-b5d9fa846d8a.jpeg)
![yield_sign](https://user-images.githubusercontent.com/20146538/31588393-09bf61bc-b1a6-11e7-90af-e63cacf65c6c.jpg)
![stop_sign](https://user-images.githubusercontent.com/20146538/31588395-0ad709ce-b1a6-11e7-8e68-e37e435c2424.jpg)
![road_work](https://user-images.githubusercontent.com/20146538/31588397-0fcb33d8-b1a6-11e7-950d-1a0f964031c4.jpg)

The first image might be difficult to classify may be because of the contour. The model was predicting left turn as keep right. It couldn't predict the left turn sign with best accuracy but it included the result in top-3. For the road work sign, it couldn't predict may be because of the image distribution. It predicted it as general caution.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| left-turn     			| Keep-right 										|
| Yield					| Yield											|
| 30 kmh	      		| 30 kmh				 				|
| Road work			| General caution     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. But, I got an accuracy of 91.1% on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in last cell of my jupyter notebook.

For the first image, the model was totally sure about the prediction (100%) and the image contained 30 kmph sign. For others, it showed very less probability.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| 30 kmph   									| 
| 1.61425255e-08    				| Speed limit (60km/h)								|
| 1.36868463e-08					| Speed limit (50km/h)											|
| 6.96060085e-11	      			| Speed limit (20km/h)			 				|
| 9.89471634e-13			    | Speed limit (80km/h)     							|


For the second image, the model was so sure about the prediction (94%) and the image did not contain Keep right sign. It contained turn left ahead which the model predicted 5% only.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .94         			| Keep right   									| 
| 5.12209758e-02    				| Turn left ahead									|
| 4.05207647e-05					| Priority road											|
| 1.51851045e-05	      			| Yield			 				|
| 4.91357969e-06				    | Speed limit (120km/h)      							|

For the third image, the model was so sure about the prediction (99%) and the image did not contain General caution sign. It contained road work sign which the model didn't predict within top-5 probabilities.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| General caution   									| 
| 2.4185338e-05    				| Right-of-way at the next intersection									|
| 1.1339109e-10					| Speed limit (30km/h)										|
| 1.1174341e-11     			| Roundabout mandatory			 				|
| 5.9223954e-14				    | Priority road      							|

For the fourth image, the model was so sure about the prediction (99%) and the image did contain Stop sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Stop									| 
| 6.8537207e-05    				| Yield								|
| 5.3735886e-05					| Speed limit (30km/h)											|
| 1.3556279e-06	      			| Speed limit (120km/h) 				|
| 1.1750384e-07			    | Go straight or right      							|

For the fifth image, the model predicted with 100% accuracy and the image did contain Yield sign. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000         			| Yield								| 
| 1.2881816e-10     				| Turn left ahead								|
| 2.8695054e-12				| Ahead only											|
| 3.8912032e-16      			| No passing			 				|
| 2.4602881e-1			    | Road work      							|
