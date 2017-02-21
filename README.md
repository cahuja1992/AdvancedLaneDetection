# Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Project Instructions
The goals / steps of this project are the following :

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

###Pipeline (single images)
#### 1. Camera Calibration
![](camera_original.png)

#### 2. Apply distortion correction to each image
![](camera_undistroted.png)

#### 3. Create a thresholded binary image
![](thresholded_binary.png)

#### 4. Perspective transform
![](bird-eye.png)

#### 5. Identify lane-line pixels and fit their positions with a polynomial
![](polynomial.png)

#### 6. Calculate the radius of curvature of the lane and the position of the vehicle with respect to the center

#### 7. Plot result back down onto tho road such that the lane area is identified clearly.
![](final_image.pngg)


###Pipeline (video)
[Video output](project_output.mp4)

###Discussion

There is some noise that is interfering with detection of lane lines,which at some instances draw higher curvature
	* Example 1: `test1.jpg`
    * Example 2: `test6.jpg`
To address the above issues the following things can be done :
	* We can increase the x gradient minimum threshold to filter out noise.
	* We can consider the sequence of images in order to undrstand the radius correctly, i.e the next frame will have similar curvature of previous frame.
