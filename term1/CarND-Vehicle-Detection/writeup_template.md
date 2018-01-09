## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.    

**Dataset used**
Here's the [vehicle dataset](https://drive.google.com/file/d/1M5HrGg0HzFymlEMfSseRWOfuIbUxpoVu/view?usp=sharing)
Here's the [non-vehicle dataset](https://drive.google.com/file/d/16LwldHg6Uaq9XNaXWAPnogVi-PmrpJqW/view?usp=sharing)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the eight code cell of the Jupyter notebook 'main.ipynb'.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of the `vehicle` and `non-vehicle` classes:

![sample_images](https://user-images.githubusercontent.com/20146538/34699691-a52d7618-f493-11e7-8c93-6b3fd660f5a9.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed some images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_visualization](https://user-images.githubusercontent.com/20146538/34659557-ce499c16-f3ee-11e7-8b7f-f5a1eb1dfc4c.png)

#### 2. Explain how you settled on your final choice of HOG parameters. Also, describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I tried various combinations of parameters and checked the time it took for feature extraction. Then, for all the parameter combinations, I trained a Linear SVM and calculated the accuracy. The code for this step is contained in the code cells nine and ten of the Jupyter notebook 'main.ipynb'. The various combinations chosen and the results are mentioned below: 

| Color Space | orient | pix_per_cell | cell_per_block | HOG channels| Time to extract HOG features(in sec) | LinearSVM Accuracy |
|:-----------:|:------:|:------------:|:--------------:|:-----------:|:----------------------------:|:------------------:|
|    RGB   |  9       | 8 | 2 | ALL | 84.15 | 0.9752
|    HSV   | 9 | 8 | 2 | ALL | 90.15 | 0.9900 |
|    LUV | 9 | 8 | 2 | ALL | 95.28 | 0.987 |
| HLS | 9 | 8 | 2 | ALL | 97.36 | 0.9899 |
|YUV | 9 | 8|2|ALL|81.51|0.9901|
|YCrCb|9|8|2|ALL|82.36|0.9789|
|YCrCb|9|16|2|ALL|51.84|0.9885|
|RGB|9|16|2|ALL|44.31|0.9555|
|HSV|9|16|2|ALL|45.17|0.9721|
|LUV|9|16|2|ALL|46.55|0.9628|
|HLS|9|16|2|ALL|47.88|0.9668|
|YUV|9|16|2|ALL|45.31|0.9716|
|YUV|11|16|2|ALL|51.2|0.9741|
|YUV|9|8|2|0|31.38|0.9535|

The parameters giving the best model accuracy were chosen. The final chosen parameters for extracting HOG features and Linear SVM are:
* colorspace = 'YUV'
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL' 
* (81.51, 'Seconds to extract HOG features...')
* (18.37, 'Seconds to train SVC...')
* ('Test Accuracy of SVC = ', 0.9901)
* (0.0021, 'Seconds to predict', 10, 'labels with SVC')

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The slide_window function takes in an image, start and stop positions and scale(overlap fraction) and returns a list of bounding boxes for the search windows, which can be used to draw boxes. The code for this step is contained in the code cell 22 of the Jupyter notebook 'main.ipynb'. Below is an illustration of the slide_window function with ystart = 400, ystop = 655 and scale = 1.5:

![window_15](https://user-images.githubusercontent.com/20146538/34698840-274fc8c6-f48f-11e7-9807-fbe0e95ebfc2.png)

The sliding window need to be of varying sizes and scale as the cars may be of different sizes based on their position w.r.t. the driving car. Following are the combination of parameters used for the final combined scale and parameter:

* ystart = 410, ystop = 510, scale = 1.0

![window_1](https://user-images.githubusercontent.com/20146538/34698865-3d554a4c-f48f-11e7-9e2e-bc01e0e00410.png)

* ystart = 425, ystop = 520, scale = 1.3

![window_13_sec](https://user-images.githubusercontent.com/20146538/34698864-3bdf7282-f48f-11e7-8c8c-97bb3b45e804.png)

* ystart = 417, ystop = 530, scale = 1.5

![window_13_sec](https://user-images.githubusercontent.com/20146538/34698931-94c61c70-f48f-11e7-980e-e736e1e9aabe.png)

* ystart = 400, ystop = 655, scale = 1.5

![window_15](https://user-images.githubusercontent.com/20146538/34698840-274fc8c6-f48f-11e7-9807-fbe0e95ebfc2.png)

* ystart = 411, ystop = 531, scale = 1.8

![window_18](https://user-images.githubusercontent.com/20146538/34698842-29f874e2-f48f-11e7-9c98-1a75fcfca09b.png)

* ystart = 401, ystop = 540, scale = 2.0

![window_2](https://user-images.githubusercontent.com/20146538/34698843-2b16c176-f48f-11e7-956a-89a206c43194.png)

* ystart = 400, ystop = 550, scale = 2.2

![window_22](https://user-images.githubusercontent.com/20146538/34698844-2ccc4784-f48f-11e7-811e-2ffc729fef2e.png)

* ystart = 400, ystop = 575, scale = 2.5

![window_25](https://user-images.githubusercontent.com/20146538/34698845-2ee39bbc-f48f-11e7-864e-4d20ef54c772.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 7 scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  The output on the six test images:

![combined_box_3](https://user-images.githubusercontent.com/20146538/34698846-30d4aad8-f48f-11e7-9e82-f94bcd84bf57.png)

![combined_box_6](https://user-images.githubusercontent.com/20146538/34698850-329176e4-f48f-11e7-8dfc-cf779e1d5861.png)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://drive.google.com/file/d/1IlWB3WsLAW1tyfSglfIX8x1neBhi1WDx/view?usp=sharing)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![heat](https://user-images.githubusercontent.com/20146538/34698855-354bf652-f48f-11e7-8800-f9e857cd80db.png)

![heat_after_thresh](https://user-images.githubusercontent.com/20146538/34698858-37585670-f48f-11e7-97bd-354fe7aa2ab1.png)

![heat_labelled](https://user-images.githubusercontent.com/20146538/34698860-3853e1b6-f48f-11e7-90d8-c886e50f4220.png)

![final_boxes](https://user-images.githubusercontent.com/20146538/34698862-39b9ac02-f48f-11e7-8bb8-7d9d0c0945f6.png)

### Here are six frames and their corresponding heatmaps:

![heat_all](https://user-images.githubusercontent.com/20146538/34699752-ee6a3672-f493-11e7-9ab9-cf481c281d29.png)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![label_all](https://user-images.githubusercontent.com/20146538/34699751-ebf11776-f493-11e7-81b6-a8de30a0d658.png)

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![final](https://user-images.githubusercontent.com/20146538/34698863-3ac2fd10-f48f-11e7-853e-c52d8b119f18.png)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* I think there can be something done to make the model identify even farther cars by collecting more data and also using even a smaller sliding window. 
* It really took a lot of time to come up with the proper parameters for HOG feature extraction, LinearSVM and also the sliding window.
* Some effort can even be made to use the detection info in the previous frame.


