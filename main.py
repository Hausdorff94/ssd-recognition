import os
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2


'''
https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
'''

def show_image(im: np.ndarray) -> None:
    '''
    Show image in a matplotlib figure

    - Parameters:
    ----------
        - im: image to show
    
    - Returns:
    ----------
        - Image: matplotlib figure
    '''
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.axis(False)
    plt.show()

def preprocess_image(path: str) -> np.ndarray:
	'''
	Pre-process the image by resizing it,
	converting it to graycale, blurring it,
	and computing an edge map.

	- Parameters:
	----------
		- path: image path

	- Returns:
	----------
		- image: pre-processed image
		- gray: grayscale image
		- blurred: blurred image
		- edged: edge detected image
	'''

	image = cv2.imread(path)
	image = imutils.resize(image.copy(), height=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 75, 200)
	
	return image, gray, blurred, edged

def find_contours(edge_im: np.ndarray) -> list:
	'''
	Find the contours in the image,
	then sort them by their size in descending order

	- Parameters:
	----------
		- edge_im: edge detected image

	- Returns:
	----------
		- cnts: list of contours
	'''

	cnts = cv2.findContours(
						edge_im.copy(),
						cv2.RETR_EXTERNAL,
						cv2.CHAIN_APPROX_SIMPLE
						)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	return cnts

def find_paper_contour(cnts: list) -> list:
	'''
	Find the contour of the paper in the image

	- Parameters:
	----------
		- cnts: list of contours

	- Returns:
	----------
		- paper_cnt: contour of the paper
	'''

	paper_cnt = None
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			paper_cnt = approx
			break

	return paper_cnt
	
def extract_display(im: np.ndarray, paper_cnt: list) -> np.ndarray:
	'''
	Extract the display from the image

	- Parameters:
	----------
		- im: image
		- paper_cnt: contour of the paper

	- Returns:
	----------
		- display: extracted display
	'''

	display = four_point_transform(im.copy(), paper_cnt.reshape(4, 2))

	return display

def cleanup_thresholded_image(warped: np.ndarray) -> np.ndarray:
	'''
	Threshold the warped image, then cleanup it
	applying a morphological transformation

	- Parameters:
	----------
		- warped: warped image

	- Returns:
	----------
		- thresh: cleaned warped image
		- kernel: kernel used for the morphological transformation
	'''

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	
	thresh = cv2.threshold(warped.copy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

	return thresh, kernel
	



# define the dictionary of digit segments so we can identify each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

# define image path
directory = 'images/'
path_dir = os.path.join(os.getcwd(), directory)
img_name = 'display2.jpeg'
img_path = os.path.join(path_dir, img_name)

image, gray, blurred, edged = preprocess_image(img_path)
cnts = find_contours(edged)
displayCnt = find_paper_contour(cnts)
warped = extract_display(gray, displayCnt)
output = extract_display(image, displayCnt)
thresh, kernel = cleanup_thresholded_image(warped)
cv2.imshow('edged', edged)
cv2.waitKey(0)

# # find contours in the thresholded image, then initialize the digit contours lists
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# digitCnts = []

# # loop over the digit area candidates
# for c in cnts:
# 	# compute the bounding box of the contour
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	# if the contour is sufficiently large, it must be a digit
# 	if w >= 15 and (h >= 30 and h <= 40):
# 		digitCnts.append(c)


# # sort the contours from left-to-right, then initialize the actual digits themselves
# digitCnts = contours.sort_contours(digitCnts,
# 	method="left-to-right")[0]
# digits = []

# # loop over each of the digits
# for c in digitCnts:
# 	# extract the digit ROI
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	roi = thresh[y:y + h, x:x + w]
	
# 	# compute the width and height of each of the 7 segments we are going to examine
# 	(roiH, roiW) = roi.shape
# 	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
# 	dHC = int(roiH * 0.05)

# 	# define the set of 7 segments
# 	segments = [
# 		((0, 0), (w, dH)),	# top
# 		((0, 0), (dW, h // 2)),	# top-left
# 		((w - dW, 0), (w, h // 2)),	# top-right
# 		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
# 		((0, h // 2), (dW, h)),	# bottom-left
# 		((w - dW, h // 2), (w, h)),	# bottom-right
# 		((0, h - dH), (w, h))	# bottom
# 	]
# 	on = [0] * len(segments)

# # loop over the segments
# 	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		
# 		# extract the segment ROI, count the total number of thresholded pixels in the segment, and then compute the area of the segment
# 		segROI = roi[yA:yB, xA:xB]
# 		total = cv2.countNonZero(segROI)
# 		area = (xB - xA) * (yB - yA)
		
# 		# if the total number of non-zero pixels is greater than 50% of the area, mark the segment as "on"
# 		if total / float(area) > 0.5:
# 			on[i]= 1
# 	# lookup the digit and draw it on the image
# 	digit = DIGITS_LOOKUP[tuple(on)]
# 	digits.append(digit)
# 	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
# 	cv2.putText(output, str(digit), (x - 10, y - 10),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# # display the digits
# #print(u"{}{}.{} \u00b0C".format(*digits))
# print(*digits)
# cv2.imshow("Input", image)
# cv2.imshow("Output", output)
# cv2.waitKey(0)

