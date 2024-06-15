"""
Module providing a function printing openCV image.

Michael Massone
5/14/2024
CS5330
Lab1
"""
##################################################################################################
## Libraries

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

##################################################################################################
# Pathing

# set file paths
BASE_DIR = Path.cwd() # get working dir
IMAGE_FILENAME = 'testimage1.png'
IMAGE_PATH = BASE_DIR / 'images' / IMAGE_FILENAME # path to image

print('Current working directory:', BASE_DIR)

##################################################################################################
## Functions

def show_image(image_path: Path, color_mode=0, duration=5000, image_name=IMAGE_FILENAME):
    ''' 
    Description: This function displays an image using openCV.

        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html

    Inputs:
            image_path (Path): Path to image.
            color_mode (int): openCV image colormode:
                                                     0 = grayscale (default)
                                                     1 = RGB
                                                    -1 = unchanged for original image
            duration (int): Time in milliseconds to display image.
            image_name (str): Name for image in results_dict, default is file name of image. 
    Outputs:
            return_dict (dict): DIctionary containing image name and cv2 image object.
                        
                                return_dict =  {'image': img}

    '''

    try:
        # create openCV image obj
        img = cv2.imread(str(image_path), color_mode)

        # check img create successfully
        if img is None:
            raise FileNotFoundError(f"No image found at {str(image_path)}.")
        
        # display image
        cv2.imshow(image_name, img)
        # duration to dispaly image
        cv2.waitKey(duration)
        # close windows
        cv2.destroyAllWindows()
        # add image to a results_dict
        result_dict = {image_name: img}

        return result_dict
    
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

    return None

##################################################################################################
## Main

def main():

    results_dict = show_image(IMAGE_PATH, image_name='image')

    if results_dict is not None:
        print("Image displayed successfully.")
    else:
        print("Failed to display the image.")
   
##################################################################################################

if __name__ == '__main__':
    
    main()