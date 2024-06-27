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
    
    # grayscale
    img1_name, img1_grayscale = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0,
                                 show=True
                                 )
    
    ## denoise with gaussian
    img1_grayscale_blur = cv2.GaussianBlur(img1_grayscale, (5, 5), 0)
    plt.imshow(img1_grayscale_blur, cmap='gray')
    plt.title("img1_grayscale_blur")
    plt.show()

    ## edge detection with canny
    img1_grayscale_blur_canny = cv2.Canny(img1_grayscale_blur, 15, 30)
    plt.imshow(img1_grayscale_blur_canny, cmap='gray')
    plt.title("img1_grayscale_blur_canny")
    plt.show()

    ## erosion and dilation
    #inv = 255 - img1_grayscale_blur_canny
    #plt.imshow(inv, cmap='gray')
    #plt.title("inverted")
    #plt.show()

    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(img1_grayscale_blur_canny, kernel, iterations=1)
    plt.imshow(img_erosion, cmap='gray')
    plt.title("erode")
    plt.show()

    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    plt.imshow(img_dilation, cmap='gray')
    plt.title('dilate')
    plt.show()
    
    blurred_edges = cv2.GaussianBlur(img_dilation, (7, 7), 0)
    plt.imshow(blurred_edges, cmap='gray')
    plt.title("blurred_edges")
    plt.show()

    ## contours
    contours, hierarchy = cv2.findContours(blurred_edges, 
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1, contours, -1, (0,255,0), 3)
    plt.imshow(img1)
    plt.show()


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()