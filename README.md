# Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
###Writeup / README

###Camera Calibration
I started with computing the camera matrix and thus the distrotion coefficients. For this, the object points on the chessboard were generated first using `cv2.findChessboardCorners`. Then the camera matrix and the distrotion coefficients were extracted using `cv2.calibrateCamera`. Then, `cv.undistrot` is used the correct the image by computed camera matrix and distrotion coefficients.
Following is the output of processed images:
![](/images/camera_calibration.png)

###Pipeline (single images)
#### 1. Corrected Distrotion-corrected image
The above computed distrition coefficients and camera matrix is used to undistrot the image.
`image = cv2.undistort(img, mtx, dist, None, mtx)` where, mtx is camera matrix and dist is distrotion coefficients.

#### Example of a Undistroted Image
![](/images/camera_undistroted.png)


#### 2. Color transforms, gradients or other methods to create a thresholded binary image
* Image has been converted to GRAY Scale using `gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`
* Then, the sobel in X which is the thresholded 
* By converting to HLS , then S channel is separated out and thresholded
* Combine the Threshold x gradient and Threshold color channel to generate the binary image

#### Example of a thresholded binary image
![](/images/thresholded_binary.png)

#### 3. Perspective transform
* Based on the manually chosen coordinated for the region of interest the mapping matrix is made for source and destination and the the `cv.getPerspectiveTransform` is applied.
* Image is the wrapped into the bird eye view using `cv2.warpPerspective`

`src = np.float32([[120, 720],[550, 470],[700, 470],[1160, 720]])
dst = np.float32([[200,720],[200,0],[1080,0],[1080,720]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)`

#### Example of a Region of Interest and Bird Eye View on the same
![](/images/bird-eye.png)

#### 4. Identify lane-line pixels and fit their positions with a polynomial
* Histogram is being generated after dividing the image into steps of equal height.
* Histogram is then smoothed
*  Left and Right Peaks are being find in the histgrams, which represents the left and right lanes.
* Then, fit the polynomial function to it.

#### Example of polynomial fitted to birds-eye-view image:
![](/images/polynomial.png)

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to the center.
Using the folowing expressions the left radius, right radius, curvature, minimum curvature and center are being calculated.
`
left_curverad = np.absolute(((1 + (2 * left_coeffs[0] * 500 + left_coeffs[1])**2) ** 1.5) /(2 * left_coeffs[0]))
right_curverad = np.absolute(((1 + (2 * right_coeffs[0] * 500 + right_coeffs[1]) ** 2) ** 1.5) /(2 * right_coeffs[0]))
curvature = (left_curverad + right_curverad) / 2
centre = center(719, left_coeffs, right_coeffs)
min_curvature = min(left_curverad, right_curverad)
`
    
#### 6. Plot result back down onto tho road such that the lane area is identified clearly.
The above calculated metrics and the fitted ploynomial are being displayed together and is done in the following code.
`   
    
    vehicle_position = vehicle_position / 12800 * 3.7
    curvature = curvature / 128 * 3.7
    min_curvature = min_curvature / 128 * 3.7

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Radius of Curvature = %d(m)' % curvature, (50, 50), font, 1, (255, 255, 255), 2)
    left_or_right = "left" if vehicle_position < 0 else "right"
    cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(vehicle_position), left_or_right), (50, 100), font, 1,
                (255, 255, 255), 2)
    cv2.putText(img, 'Min Radius of Curvature = %d(m)' % min_curvature, (50, 150), font, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Left poly coefficients = %.3f %.3f %.3f' % (left_coeffs[0], left_coeffs[1], left_coeffs[2]), (50, 200), font, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Right poly coefficients = %.3f %.3f %.3f' % (right_coeffs[0], right_coeffs[1], right_coeffs[2]), (50, 250), font, 1, (255, 255, 255), 2)

#### Example of lane boundaries and numerical estimation of lane curvature and vehicle position put together
![](/images/final_image.png)


###Pipeline (video)
[Video output](/project_output.mp4)

###Discussion

There is some noise that is interfering with detection of lane lines,which at some instances draw higher curvature that can be seen in test2.jpg and test5.jpg.
To address the above issues the following things can be done :
	
	* We can increase the x gradient minimum threshold to filter out noise.
	* We can consider the sequence of images in order to understand the radius correctly, i.e the next frame will have similar curvature of previous frame.
	* Thresholds choosed to create binary image is just by trial and error approach which is not robust, thus some ggod method needs to be evaluated.
	* Source and destinations matrix for masking binary thresholded images is to be tunned to better generalization.
