## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[experiment]: ./output_images/experiment.png
[hog]: ./output_images/hog.png
[pipeline]: ./output_images/pipeline.png
[result]: ./output_images/result.png
[h]: ./output_images/h.png
[l]: ./output_images/l.png
[s]: ./output_images/s.png
[box]: ./output_images/box.png
[heat0]: ./output_images/heat0.png
[heat1]: ./output_images/heat1.png
[heat2]: ./output_images/heat2.png
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the part 1 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces for the hog algorithm.  

Here are the HOG result on all three of HLS color channel.

![alt text][h]

![alt text][l]

![alt text][s]

Here is an example using the grayscale color and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, it seems that grayscale value is the best one that capture the visual information of the object. I then try to run classifier on some random combination of other color space and parameter, and I ended up decided to use grayscale image with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM using only HOG features, because with the HOG features, I was able to get 99.1% of accuracy, but the computing time is already slow (about 1 frame per second). So I decided to use HOG as the only feature.

The code is in the 2nd part of the ipython notebook. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Here are the parameters of different scale of searching windows I used.

```python
ystart=400, rows=1, scale=1.3
ystart=400, rows=1, scale=1.9
ystart=400, rows=1, scale=2.5
ystart=400, rows=1, scale=3.6
```

I found that all vehicle roof is about the same height in the image, which is 400. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using grayscale HOG features as the feature vector, which provided a nice result.  Here are some example images:

![alt text][box]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./vehicle_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][heat0]
![alt text][heat1]
![alt text][heat2]

### Here is a video demo of heatmaps:

Here's a [link to heatmap video](./heatmap.mp4)

### Here are the 3 stages of the pipeline, Generating bounding boxes of all matching features, generate heat map, and creating final bounding box for the detection. (The third and fourth image are reversed)

![alt text][pipeline]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The most difficutl problem I faced in this project is to handle false positive. A higher threshold will result in bad detection on vehicle, but a lower threshold will result in false positive.

To solved this problem, I maintain a heatmap across different frames. I took the heatmap from current frame, substracted a fixed value from it, so that the system will remember where is the vehicle on last frame. Then I applied a `cv2.GaussianBlur()` to the image, because the posibility of the new position is not exactlly at the old position, it could move a little bit from the old position, which can be represent by gaussian blur.

My pipeline will likely fail when there are some huge vehicle on the road such as truck or train. To make it more robust, I'll need to train the classifier on more complex dataset and fine tune the system on different videos.
