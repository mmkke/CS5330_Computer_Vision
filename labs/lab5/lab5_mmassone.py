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
IMAGE_FILENAME_1 = 'dark_image3'
FILE_EXT = '.jpeg'
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
def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")

    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)


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
                                 show=True
                                 )
    
    # grayscale
    img1_name, img1_grayscale = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0,
                                 show=False
                                 )
    


    ## Gamma
    img1_gamma = adjust_gamma(img1_grayscale, 2.2)
    #plt.imshow(img1_gamma, cmap = 'gray', vmin=0, vmax=255)
    #plt.title('Gamma Adjusted')
    #plt.show()

    # bilateral filter
    img1_BL = cv2.bilateralFilter(img1_gamma, d=9, sigmaColor=75, sigmaSpace=75)
    #plt.imshow(img1_BL, cmap = 'gray', vmin=0, vmax=255)
    #plt.title('Bilateral')
    #plt.show()

    ## Laplacian 
    img1_lap = cv2.Laplacian(img1_BL, ddepth=cv2.CV_8U, ksize=5)
    print(img1_lap)
    #plt.imshow(img1_lap, cmap = 'gray')
    #plt.title('Laplacian')
    #plt.show()

    # color image
    img1_color = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    img1_color[:, :, 1] = img1_lap
    #plt.imshow(img1_color, vmin=0, vmax=255)
    #plt.title('Color')
    #plt.show()

    ## Canny
    img1_canny = cv2.Canny(img1_BL, 15, 15)
    #plt.imshow(img1_canny, cmap = 'gray', vmin=0, vmax=255)
    #plt.title('Canny')
    #plt.show()

    # color image
    img1_color = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    img1_color[:, :, 1] = img1_canny
    #plt.imshow(img1_color, vmin=0, vmax=255)
    #plt.title('Color')
    #plt.show()

    # mix original grayscale image with edge detection image
    # convert grayscale to 3 channels
    gray_image_3ch = cv2.cvtColor(img1_gamma, cv2.COLOR_GRAY2BGR)

    # blendimages
    alpha = 0.4
    blended_image = cv2.addWeighted(gray_image_3ch, 1 - alpha, img1_color, alpha, 0)
    plt.imshow(blended_image, vmin=0, vmax=255)
    plt.title('Blended')
    plt.show()



##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()