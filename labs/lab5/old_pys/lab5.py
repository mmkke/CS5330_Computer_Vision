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


def apply_unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def morphological_operations(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    return dilated
##################################################################################################
## Kernels
##################################################################################################

sobel_x = (1/9) * np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]
                           ])

sobel_y = (1/9) * np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]
                           ])

##################################################################################################
## Main
##################################################################################################
def main():

    imgs = []

    ## Read in Images
    # img1
    img1_name, img1 = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0
                                 )
    
    plt.imshow(img1, cmap = 'gray')
    plt.title('Original')
    plt.show()

    ## Gaussian Blur
    img1_blur = cv2.GaussianBlur(img1, (3, 3), 0)
    plt.imshow(img1_blur, cmap = 'gray')
    plt.title('Gaussian Blur')
    plt.show()

    print(img1_blur)
    
    ## Brighten with a look up table
    # create look up table for increasing brightness
    lut = np.array([min(255, i + 100) for i in range(256)], dtype=np.uint8)

    # Apply the LUT to the image
    img1_lut = cv2.LUT(img1_blur, lut)
    print(img1_lut)

    plt.imshow(img1_lut, cmap = 'gray', vmin=0, vmax=255)
    plt.title('LUT')
    plt.show()


    ## Histogra Equalization
    img1_eq = cv2.equalizeHist(img1_blur)
    plt.imshow(img1_eq, cmap = 'gray', vmin=0, vmax=255)
    plt.title('EQ')
    plt.show()

    ##CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img1_clahe = clahe.apply(img1_eq)
    plt.imshow(img1_clahe, cmap = 'gray', vmin=0, vmax=255)
    plt.title('CLAHE')
    plt.show()

    ## Gaussian Blur
    img1_blur = cv2.GaussianBlur(img1_clahe, (3, 3), 0)
    plt.imshow(img1_blur, cmap = 'gray')
    plt.title('Gaussian Blur')
    plt.show()


    ## Canny Edge Detection

    img1_canny = cv2.Canny(img1_blur, 50, 50)
    plt.imshow(img1_canny, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Canny')
    plt.show()

    ## Laplacian of Gaussian

    img1_lap = cv2.Laplacian(img1_blur, ddepth=cv2.CV_8U)
    plt.imshow(img1_lap, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Laplacian')
    plt.show()

    ## Gamma

    img1_gamma = adjust_gamma(img1_blur, 1.0)
    plt.imshow(img1_gamma, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Gamma Adjusted')
    plt.show()

    ## Laplacian of Gamma Adjusted
    img1_lap = cv2.Laplacian(img1_gamma, ddepth=cv2.CV_8U, ksize=3)
    plt.imshow(img1_lap, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Laplacian')
    plt.show()

    ##CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img1_clahe = clahe.apply(img1)
    plt.imshow(img1_clahe, cmap = 'gray', vmin=0, vmax=255)
    plt.title('CLAHE')
    plt.show()

    ## Median Blur
    img1_blur  = cv2.medianBlur(img1_clahe, ksize=3)
    plt.imshow(img1_blur, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Median')
    plt.show()

    ## Unsharp Mask
    img1_usharp = apply_unsharp_mask(img1_blur)
    plt.imshow(img1_usharp, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Unsharp')
    plt.show()


    ## Canny
    img1_canny = cv2.Canny(img1_usharp, 50, 50)
    plt.imshow(img1_canny, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Canny')
    plt.show()


    ## Gamma
    img1_gamma = adjust_gamma(img1, 2.2)
    plt.imshow(img1_gamma, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Gamma Adjusted')
    plt.show()

    # bilateral filter
    img1_BL = cv2.bilateralFilter(img1_gamma, d=9, sigmaColor=75, sigmaSpace=75)
    plt.imshow(img1_BL, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Bilateral')
    plt.show()

   ## Laplacian
    img1_lap = cv2.Laplacian(img1_BL, ddepth=cv2.CV_8U)
    plt.imshow(img1_lap, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Laplacian')
    plt.show()

    ## Canny
    img1_canny = cv2.Canny(img1_BL, 25, 25)
    plt.imshow(img1_canny, cmap = 'gray', vmin=0, vmax=255)
    plt.title('Canny')
    plt.show()


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()