##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Train a classifier Linear SVM classifier and test it on the testing set.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image2]: ./output_images/car_1.png
[image21]: ./output_images/not_car_1.png
[image3]: ./output_images/test4_boxes.png
[image4]: ./output_images/test6_new.png
[image41]: ./output_images/test5_new.png
[image42]: ./output_images/test4_new.png
[image5]: ./output_images/test1_new1.png
[image51]: ./output_images/test1_heat1.png
[video1]: ./project_video_annotated.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. HOG features extraction from the training images.

First I defined parameters for the HOG, spatial and histogram features in the lines 75 through 85 in the `detection_pipeline.py`.

Then I continued by reading in all the `vehicle` and `non-vehicle` images manually divided into training and testing sets (80% and 20% respectively) lines 208 through 249 of the file called `feature_extraction.py` and converting them to feature arrays getting HOG, spatial and histogram features in lines 14 through 143 in the `feature_extraction.py`.

Here is an example using the `HLS` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image21]

####2. Choice of HOG parameters.

During the last Udacity lesson I've been experimenting with different parameters and got to some conclusions. I liked LUV, HLS detection capabilities, but still there were some issues.
But that was not enough since I hadn't been table to test all the combinations.
That's why I continued to experiment during the implementation and get to the conclusion that parameters in lines 75 through 85 in `detection_pipeline.py` provide me best
recognition results.

####3. Classifier training

I trained a linear SVM as was suggested and it is located in lines 108 through 118 `detection_pipeline.py`. 
Before training itself - I've normalised the data and shuffled them in lines 91 through 104 `detection_pipeline.py`.

###Sliding Window Search

####1. Sliding window search.

I decided to search for cars in the lower part of the image (there are no vehicles in the air or in the trees anyway). Next thing I've decided to use different dimensions
for the sliding windows and for each size of the sliding windows I selected the corresponding area to search for the car (the further - the smaller). 

The sliding windows
approach is located in lines 201 through 246 `search_windows.py`, different scale search is located in the same file in lines 145 through 198 and
basic search window function is in lines 45 through 87.

![alt text][image3]

####2. Examples of test images to demonstrate the pipeline.

I've used the the pipeline in the lines 13 through 44 in `detection_pipeline.py`. Basically it worked fine when I increased the heat threshold. 

Here are some example images:

![alt text][image4]
![alt text][image41]
![alt text][image42]
---

### Video Implementation

####1. Link to the annotated video
Here's a [link to my video result](./project_video_annotated.mp4)


####2. Filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.
Code is located in lines 63 through 70 in the `detection_pipeline.py` and implementation is in lines 90 through 120 in the `search_windows.py`.

In addition to that I've implemented the class in order to store the information about previously detected vehicle positions (heated windows) and used that information in
order to sort out false positives (lines 62 through 65 `detection_pipeline.py` and lines 9 through 42 in `search_windows.py`).

### Here is the example showing the initially windows and resulting labelled ones:

![alt text][image5]

And here is labelled one based on heat threshold:
![alt text][image51]

---

###Discussion

#### Problems and Issues

Well, to start with, in the annotated video there are some false positives in the left part of the screen (possibly the cars from the other direction), but in real life having
any kind of false positives is not acceptable, since it is crucial for the car to know the positions of other cars.

Second of all, the positions of the car should be extremely accurate, since this is really import either. (e.g. bounding box around the car could actually block the part
of the current vehicle lane leading to unnecessary breaking or maneuvering).

The last thing I would like to mention is that, the there are different kind of vehicles on the road, so we need to track basically everything around us. That means
that we need to train the vehicle to recognise trucks, humans, animals, even maybe planes (what if it somehow landed on the road). 

So we probably need to use not only the computer vision with machine learning, but also some kind of different sensors to generalise different things that could potentially block the road.

