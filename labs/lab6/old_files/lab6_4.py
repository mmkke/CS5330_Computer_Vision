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
    img1_name_grscl, img1_grscl = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0,
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
    img1_blur = cv2.bilateralFilter(img1_grscl, d=5, sigmaColor=50, sigmaSpace=50)
    plt.imshow(img1_blur, cmap='gray')
    plt.title("bilateral")
    plt.show()
    
    adaptive_thresh_gaussian = cv2.adaptiveThreshold(img1_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 11, 2)
    plt.imshow(adaptive_thresh_gaussian, cmap='gray')
    plt.title("adaptive_thresh_gaussian")
    plt.show()

    ## denoise color image with media filter
    img1_blur = cv2.medianBlur(adaptive_thresh_gaussian, 5)
    plt.imshow(img1_blur, cmap='gray')
    plt.title("median")
    plt.show()
  

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(adaptive_thresh_gaussian, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing, cmap='gray')
    plt.title("closing 5x5")
    plt.show()  


    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray')
    plt.title("opening 3x3")
    plt.show()   

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray')
    plt.title("opening 5x5")
    plt.show()  

    kernel = np.ones((7,7), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray')
    plt.title("opening 7x7")
    plt.show()  

    kernel = np.ones((9,9), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray')
    plt.title("opening 9x9")
    plt.show() 




##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()