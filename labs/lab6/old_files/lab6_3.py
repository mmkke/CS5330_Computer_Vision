"""
Contours and Shadows

Michael Massone
6/26/2024
CS5330
Lab 6

Instructions: 

Finding the contour of an object is the first step to object detection. How-
ever, lighting and shadows can cause a lot of issues when trying to find
contours. Especially with lighter color objects. In this lab you must find a
way to deal with and mitigate the effect of shadows on an object. There are
six colored blocks, your must write a program to find the contour and center
point of all six objects.
"""

##################################################################################################
## Libraries
##################################################################################################

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

##################################################################################################
# Pathing
##################################################################################################

# set file paths
BASE_DIR = Path.cwd() # get working dir
IMAGE_FILENAME_1 = 'blocks'
FILE_EXT = '.JPG'
IMAGE_PATH = BASE_DIR / 'images' # path to images

print('Current working directory:', BASE_DIR)


##################################################################################################
## Modules
##################################################################################################
# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent.parent

# Add the parent directory to sys.path
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import cv_utils as cv

##################################################################################################
## Functions
##################################################################################################



##################################################################################################
## Main
##################################################################################################
def main():

    imgs = []

    ## Read in Image
    img1_name, img1 = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=1,
                                 bgr=False,
                                 show=False
                                 )
    
    ## Display each channel in grayscale
    # Check if the image is loaded correctly
    if img1 is None:
        print("Error: Could not load image.")
    else:
        # Split the image into channels
        blue_channel, green_channel, red_channel  = cv2.split(img1)

        # Display images
        plt.figure(figsize=(10, 7))

        plt.subplot(2, 2, 1)
        plt.imshow(img1)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(blue_channel, cmap='gray')
        plt.title('Blue Channel')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(green_channel, cmap='gray')
        plt.title('Green Channel')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(red_channel, cmap='gray')
        plt.title('Red Channel')
        plt.axis('off')

        plt.show()
    
    
    ## denoise color image with bilateral filter
    img1_blur = cv2.bilateralFilter(img1, d=15, sigmaColor=150, sigmaSpace=50)
    plt.imshow(img1_blur, cmap='gray')
    plt.title("img1_blur")
    plt.show()

    ## EDGE DETECTION
    # edge detection with canny on each color channel in grayscale
    # take max (maybe mean?) values from each channel's grayscale image
    canny = np.max( np.array(
                            [ cv2.Canny(img1_blur[:,:, 0], 25, 45), 
                              cv2.Canny(img1_blur[:,:, 1], 25, 45), 
                              cv2.Canny(img1_blur[:,:, 2], 25, 45) ]), 
                              axis=0)
    
    # reduce noise with thresholding
    mean = np.mean(canny)
    canny = np.where(canny <= mean, 0, canny)
    plt.imshow(canny, cmap='gray')
    plt.title("canny")
    plt.show()

    ## smooth with bilateral filter
    edges_blur = cv2.bilateralFilter(canny.astype(np.uint8), d=10, sigmaColor=100, sigmaSpace=100)
    ## denoise with median filter
    #edges_blur = cv2.medianBlur(canny.astype(np.uint8), 3)
    ## denoise with gaussain
    #edges_blur = cv2.GaussianBlur(edges.astype(np.uint8), (3, 3), 0)
    plt.imshow(edges_blur, cmap='gray')
    plt.title("edges_blur")
    plt.show()


    ## MORPHOLOGICAL OPERATIONS
    # dilation
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(edges_blur.astype(np.uint8), kernel, iterations=3)
    plt.imshow(img_dilation, cmap='gray')
    plt.title('dilate')
    plt.show()

    ## erosion
    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(img_dilation.astype(np.uint8), kernel, iterations=1)
    plt.imshow(img_erosion, cmap='gray')
    plt.title("erode")
    plt.show()

    
    ## COUNTOURS
    contours, hierarchy = cv2.findContours(img_erosion.astype(np.uint8), 
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    # Print hierarchy information for debugging
    print("Number of contours found:", len(contours))

    # Smooth contours using approxPolyDP
    epsilon = 0.01 * cv2.arcLength(contours[0], True)
    smoothed_contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
    print("Number of smoothed contours found:", len(smoothed_contours))

    # Draw original and smoothed contours
    img_contours = img1.copy()
    img_smoothed_contours = img1.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(img_smoothed_contours, smoothed_contours, -1, (0, 255, 0), thickness=3)

    # Display the results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_contours)
    plt.title('Original Contours')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_smoothed_contours)
    plt.title('Smoothed Contours')
    plt.axis('off')

    plt.show()

    ## Draw smoothed contours with centroids
    # Loop over each contour
    for contour in smoothed_contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        
        # Calculate the centroid using the moments
        if M["m00"] != 0:  # avoids division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw the contours and centroids
        cv2.drawContours(img_smoothed_contours, [contour], -1, (0, 255, 0), 2)
        cv2.circle(img_smoothed_contours, (cX, cY), 10, (0, 0, 0), -1)
        cv2.putText(img_smoothed_contours, "centroid", (cX - 30, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)

    # Final result
    plt.figure(figsize=(10, 5))
    plt.imshow(img_smoothed_contours)
    plt.title('Contours and Centroids')
    plt.axis('off')

    plt.savefig('images/final_image.png')
    plt.show()


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()