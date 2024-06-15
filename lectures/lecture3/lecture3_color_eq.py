"""
Lecture 3 Module

Michael Massone
5/22/2024
CS5330
"""

##################################################################################################
## Libraries
##################################################################################################

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

##################################################################################################
# Pathing
##################################################################################################

# set file paths
BASE_DIR = Path.cwd() # get working dir
IMAGE_FILENAME = 'flowers'
FILE_EXT = '.jpg'
IMAGE_PATH = BASE_DIR / 'images' # path to images

print('Current working directory:', BASE_DIR)


##################################################################################################
## Functions
##################################################################################################
def show_image(image_path: Path, color_mode=0, image_name=IMAGE_FILENAME):
    ''' 
    Description: This function displays an image using openCV.

        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html

    Inputs:
            image_path (Path): Path to image.
            color_mode (int): openCV image colormode:
                                                     0 = grayscale (default)
                                                     1 = RGB
                                                    -1 = unchanged for original image
            image_name (str): Name for image in results_dict, default is file name of image. 
    Outputs:
            return_dict (dict): Dictionary containing image name and cv2 image object.
                        
                                return_dict =  {'image': img}

    '''

    try:
        # Create openCV image obj
        img = cv2.imread(str(image_path), color_mode)

        # Dheck img create successfully
        if img is None:
            raise FileNotFoundError(f"No image found at {str(image_path)}.")
        
        # Display image
        if color_mode != 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 (BGR) -> matplotlib (RGB)
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(image_name)
        plt.show()

        result_dict = {image_name: img}
        return result_dict
    
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None


##################################################################################################
## Main
##################################################################################################
def main():

    # Get original image
    img_color = cv2.imread('images/flowers.jpg', 1)

    cv2.imshow('Original Image', img_color)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    img_grayscale = cv2.imread('images/flowers.jpg', 0)
    
    plt.hist(img_color.ravel(), bins=256, range=[0, 256], label='original')
    plt.hist(img_grayscale.ravel(), bins=256, range=[0, 256], label='grayscale')
    plt.legend()
    plt.title('Color vs Grayscale Histogram (single channel)')
    plt.show()

    # using cv2 histogram for color images
    hist_blue = cv2.calcHist([img_color], [0], None, [256], [0, 256]) # channel=[0], slecting for blue channel
    hist_green = cv2.calcHist([img_color], [1], None, [256], [0, 256]) # channel=[1], slecting for green channel
    hist_red = cv2.calcHist([img_color], [2], None, [256], [0, 256]) # channel=[2], slecting for red channel

    plt.plot(hist_blue, color = 'b')
    plt.plot(hist_green, color = 'g')
    plt.plot(hist_red, color = 'r')
    plt.title('Original Histogram')
    plt.show()

    ## equalize 
    img_equalize_b = cv2.equalizeHist(img_color[:, :, 0])
    img_equalize_g = cv2.equalizeHist(img_color[:, :, 1])
    img_equalize_r = cv2.equalizeHist(img_color[:, :, 2])

    img_equalize = np.stack([img_equalize_b, img_equalize_g, img_equalize_r], axis=-1)

    cv2.imshow('Equalized Image', img_equalize)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    # using cv2 histogram for equalized color images
    hist_blue = cv2.calcHist([img_equalize], [0], None, [256], [0, 256]) # channel=[0], slecting for blue channel
    hist_green = cv2.calcHist([img_equalize], [1], None, [256], [0, 256]) # channel=[1], slecting for green channel
    hist_red = cv2.calcHist([img_equalize], [2], None, [256], [0, 256]) # channel=[2], slecting for red channel

    plt.plot(hist_blue, color = 'b')
    plt.plot(hist_green, color = 'g')
    plt.plot(hist_red, color = 'r')
    plt.title('Equalized Histogram')
    plt.show()


##################################################################################################
## END
##################################################################################################

if __name__ == '__main__':
    main()