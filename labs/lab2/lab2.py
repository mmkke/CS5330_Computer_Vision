"""
Module providing functions for grayscaling images using the Average and NTSC Method.

Michael Massone
5/22/2024
CS5330
Lab2
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
def grayscale_avg_method(results_dict, image_name=IMAGE_FILENAME):
    '''
    Description:
        Convert an image to grayscale using the Average method.

        The grayscale value is calculated using the formula:

        grayscale_value = (r + g + b)/3

        where `r`, `g`, and `b` are the red, green, and blue channel values of a pixel. 

    Inputs:
        results_dict (dict): {image_name: img}
        image_name (str): name of image to be converted to grayscale

    Outputs:
        results_dict (dict): {image_name: img} with addition of new grayscale image

    '''
    # Retrieve image value
    img = results_dict[image_name]
    # Apply AVG Method
    img_grayscale_avg_method = np.sum(img, axis=2) / 3
    # Add to dict
    results_dict[image_name + "_grscl_avg"] = img_grayscale_avg_method

    return results_dict

##################################################################################################
def grayscale_nstc_method(results_dict, image_name=IMAGE_FILENAME):
    '''
    Description:
        Convert an image to grayscale using the NTSC method.

        The grayscale value is calculated using the formula:

        grayscale_value = (0.299 * r) + (0.587 * g) + (0.114 * b)

        where `r`, `g`, and `b` are the red, green, and blue channel values of a pixel. 

    Inputs:
        results_dict (dict): {image_name: img}
        image_name (str): name of image to be converted to grayscale

    Outputs:
        results_dict (dict): {image_name: img} with addition of new grayscale image

    '''
    # Define weights for NSTC formula
    weights = np.array([0.114, 0.587, 0.299])
    # Retrieve image value
    img = results_dict[image_name]
    # APply NSTC method
    img_weighted_bgr = img * weights.reshape(1, 1, 3)
    img_grayscale_nstc_method = np.sum(img_weighted_bgr, axis=2)

    # Add to dict
    results_dict[image_name + "_grscl_ntsc"] = img_grayscale_nstc_method
    return results_dict


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

    # Convert original image to grayscale using avg and nstc methods
    results_dict = grayscale_avg_method(results_dict)
    results_dict = grayscale_nstc_method(results_dict)

    # Replace original image with grayscale
    del results_dict[IMAGE_FILENAME]
    img_grscl_native_method = cv2.imread(str(IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT)), cv2.IMREAD_GRAYSCALE)
    results_dict[IMAGE_FILENAME + '_grscl_native'] = img_grscl_native_method

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