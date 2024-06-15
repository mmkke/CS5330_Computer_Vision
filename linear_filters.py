"""
Module providing functions for linear filtering images using various kernels.

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
## Pathing
##################################################################################################

# set file paths
BASE_DIR = Path.cwd() # get working dir
IMAGE_FILENAME = 
FILE_EXT = '.jpg'
IMAGE_PATH = BASE_DIR / 'images' # path to images

print('Current working directory:', BASE_DIR)

##################################################################################################
## Parameters
##################################################################################################

kernel = None

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
def correlation_filter(img, kernel):
    pass

##################################################################################################
def convolution_filter(img, kernel):
    pass


##################################################################################################
## Main
##################################################################################################
def main():

    # Get original image
    results_dict = show_image(IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT), color_mode=1)

    # Check that image was stored in dict
    if results_dict is not None:
        print("Image displayed successfully.")
    else:
        print("Failed to display the image.")
        return

    # Run filters
    results_dict = correlation_filter()

    results_dict = convolution_filter()

    # Add images to list
    img_list = []
    for name, img in results_dict.items():
        img_list.append((name, img))

    # Display images
    plt.figure(figsize=(15, 5))

    # Original Image in CV2 grayscale
    plt.subplot(1, 3, 1)  
    plt.imshow(img_list[0][1], cmap='gray')
    plt.title(img_list[0][0])
    plt.colorbar()  

    # Image using AVG Method
    plt.subplot(1, 3, 2)  
    plt.imshow(img_list[1][1], cmap='gray')
    plt.title(img_list[1][0])
    plt.colorbar()  

    # Image using NSTC Method
    plt.subplot(1, 3, 3)  
    plt.imshow(img_list[2][1], cmap='gray')
    plt.title(img_list[2][0])
    plt.colorbar() 

    # Figure attributes
    plt.tight_layout()
    plt.savefig(IMAGE_PATH / 'grayscale_comparison.png')
    plt.show()
   
##################################################################################################
## END
##################################################################################################

if __name__ == '__main__':
    main()