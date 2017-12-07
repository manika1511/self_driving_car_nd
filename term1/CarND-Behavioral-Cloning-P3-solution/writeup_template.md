# Behavioral Cloning

In this project, deep neural networks and convolutional neural networks are used to clone driving behavior. A model is trained, validated and tested using Keras. The model outputs a steering angle to an autonomous vehicle.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I have used the NVIDIA architecture for training the model. This model consists of 5 convolutional layers, one flatten layer and four dense layers. In addition to these layers, there is a Lambda layer for image normalization and a cropping layer to crop the image to just include the road area and remove the surrounding.

Original Image
![originalimage](https://user-images.githubusercontent.com/20146538/32481374-ea4498c6-c347-11e7-82ef-494b3b17d5fe.png)

Cropped Image
![croppedimage](https://user-images.githubusercontent.com/20146538/32481376-eb368988-c347-11e7-841c-f41e99922582.png)

#### 2. Attempts to reduce overfitting in the model

The model was trained by splitting the dataset into train and validation to avoid overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The number of epochs were tuned. The epochs were kept till which the error reduced. So, here epochs were 3.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I tried to collect the data by running the simulator on the road center for two laps, then recovering from left and right for one lap and then one laps moving in opposite direction for one lap. I also tried to get quality data by running a lap in the second track. But, I was unable to get Quality data. So, finally I chose the dataset provided by Udacity. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first normalize the input image and then crop it.

I used the Nvidia architecture and added the normalization and cropping layer in the start. This model consists of 5 convolutional layers, one flatten layer and four dense layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used the entire dataset. The center image and also the left and right images with steering correction of 0.2. I tried without using the left and right images, but the model didn't work well at steep turns. After adding the left and right images, the model improved and worked well on the turns.

To combat the overfitting, I split the dataset into train and test and then gauged the performance of the model. After gauging the performance, it was found that the model was not overfitting or underfitting.

Then I ran the model and started with epohs=10. The loss first decreased but then started increasing. I found that till 3 epochs the loss decreased and then it increased. So, again I ran for 3 epochs and the model got trained as expected. 

The final step was to run the simulator to see how well the car was driving around track one. Initially, with the data I collected on my own, the vehicle ran well on normal truns but went off the track on the steeps ones.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

<img width="874" alt="screen shot 2017-11-06 at 10 51 46 pm" src="https://user-images.githubusercontent.com/20146538/32481871-10e4df98-c34a-11e7-96f3-6371e0554036.png">

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![originalimage](https://user-images.githubusercontent.com/20146538/32481374-ea4498c6-c347-11e7-82ef-494b3b17d5fe.png)
 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it slips of the road. Then, I drove one lap in the opposite direction. 
But, I was unable to get Quality data. So, finally I chose the dataset provided by Udacity.

To augment the data sat, I also flipped images and angles thinking that this would help generating samples, if the vehicle travelled in the anti-clockwise direction. For example, here is an image that has then been flipped:

Original Image
![ol](https://user-images.githubusercontent.com/20146538/32482130-26c012c8-c34b-11e7-9734-90a2e0927f33.png)

Flipped Image
![fl](https://user-images.githubusercontent.com/20146538/32482140-2c31db42-c34b-11e7-8196-302eeb0a2416.png)

After the collection process, I had 24108 number of data points. I then preprocessed this data by normalizing it and then cropping it to just see the data actually needed.

Original Image
![originalimage](https://user-images.githubusercontent.com/20146538/32481374-ea4498c6-c347-11e7-82ef-494b3b17d5fe.png)

Cropped Image
![croppedimage](https://user-images.githubusercontent.com/20146538/32481376-eb368988-c347-11e7-841c-f41e99922582.png)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced. I used an adam optimizer so that manually training the learning rate wasn't necessary.
