"""
Store Brand Night Vision

Michael Massone
6/12/2024
CS5330
Lab 5

Instructions: 

We have covered various image processing techniques throughout this semester.
Take all of those techniques and try to recreate night vision military technol-
ogy. While we do not have access to the cameras that the military uses, we
do have ways to process images that can get similar results. This assignment
is meant to be fully open ended. You can use any pre-built CV2 code.
Even ones that we have not covered in class.
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
def sobelFilter (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_64F, 1, 0, 3)
    sobelY = cv2.Sobel(channel, cv2.CV_64F, 0, 1, 3)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255
    return sobel


##################################################################################################
## Main
##################################################################################################
def main():

    imgs = []

    ## Read in Images
    # color
    img1_name, img1 = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=1,
                                 bgr=False,
                                 show=True
                                 )
    
    # grayscale
    img1_name, img1_grayscale = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0,
                                 show=False
                                 )
    # display each channel in grayscale
    # Check if the image is loaded correctly
    if img1 is None:
        print("Error: Could not load image.")
    else:
        # Split the image into its B, G, R channels
        blue_channel, green_channel, red_channel  = cv2.split(img1)

        # Display the original and the channels
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
    
    
    ## denoise with bilateral filter
    img1_blur = cv2.bilateralFilter(img1, d=15, sigmaColor=150, sigmaSpace=50)
    plt.imshow(img1_blur, cmap='gray')
    plt.title("img1_blur")
    plt.show()

    ## edge detection with sobel

    
    # take max (maybe mean?) values from each channel's grayscale image
    edges = np.mean( np.array(
                            [ sobelFilter(img1_blur[:,:, 0]), 
                              sobelFilter(img1_blur[:,:, 1]), 
                              sobelFilter(img1_blur[:,:, 2]) ]), 
                              axis=0 )
    mean = np.mean(edges)
    #edges = np.where(edges <= mean, 0, edges)
    plt.imshow(edges, cmap='gray')
    plt.title("edges")
    plt.show()

        ## edge detection with canny
    # take max (maybe mean?) values from each channel's grayscale image
    canny = np.max( np.array(
                            [ cv2.Canny(img1_blur[:,:, 0], 25, 45), 
                              cv2.Canny(img1_blur[:,:, 1], 25, 45), 
                              cv2.Canny(img1_blur[:,:, 2], 25, 45) ]), 
                              axis=0 )
    mean = np.mean(canny)
    canny = np.where(canny <= mean, 0, canny)
    plt.imshow(canny, cmap='gray')
    plt.title("canny")
    plt.show()

    ## denoise with bilateral filter
    #edges_blur = cv2.bilateralFilter(edges.astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25)

    ## denoise with median filter
    #edges_blur = cv2.medianBlur(edges.astype(np.uint8), 3)

    ## denoise with gaussain
    edges_blur = cv2.GaussianBlur(canny.astype(np.uint8), (3, 3), 0)
    plt.imshow(edges_blur, cmap='gray')
    plt.title("edges_blur")
    plt.show()

    ret, thresh_binary = cv2.threshold(edges_blur, 12, 255, cv2.THRESH_BINARY)
    ret, thresh_binary_inv = cv2.threshold(edges_blur, 50, 255, cv2.THRESH_BINARY_INV)
    ret, thresh_trunc = cv2.threshold(edges_blur, 10, 255, cv2.THRESH_TOZERO)
    thresh_adaptive_mean = cv2.adaptiveThreshold(edges_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh_adaptive_gaussian = cv2.adaptiveThreshold(edges_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    plt.imshow(thresh_binary, cmap='gray')
    plt.title('thresh_binary')
    plt.axis('off')
    plt.show()

    ## erode
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(thresh_binary.astype(np.uint8), kernel, iterations=1)
    plt.imshow(img_erosion, cmap='gray')
    plt.title("erode")
    plt.show()

    # dilation
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(img_erosion.astype(np.uint8), kernel, iterations=5)
    plt.imshow(img_dilation, cmap='gray')
    plt.title('dilate')
    plt.show()

    ## thresholding
    # histogram
    plt.hist(img_dilation.ravel(), 256, (0, 255))
    plt.scatter(np.mean(img_dilation), 0, s=100, c='red')
    plt.show()
    #img_threshold = np.where(img_dilation < 40, 0, 255)
    #plt.imshow(img_threshold, cmap='gray')
    #plt.title('threshold')
    #plt.show()


    
    ## contours
    contours, hierarchy = cv2.findContours(img_dilation.astype(np.uint8), 
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
    # Print hierarchy information for debugging
    print("Number of contours found:", len(contours))
    print("Hierarchy:", hierarchy)
    #img1 = cv2.drawContours(img1, contours, -1, (0,255,0), 3)
    #plt.imshow(img1)
    #plt.show()

    # Smooth contours using approxPolyDP
    epsilon = 0.01 * cv2.arcLength(contours[0], True)
    smoothed_contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]

    # Draw original and smoothed contours
    img_contours = img1.copy()
    img_smoothed_contours = img1.copy()

    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(img_smoothed_contours, smoothed_contours, -1, (0, 255, 0), thickness=cv2.FILLED)

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


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()