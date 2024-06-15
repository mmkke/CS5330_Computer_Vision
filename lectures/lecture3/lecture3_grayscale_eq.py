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
IMAGE_FILENAME = 'dog'
FILE_EXT = '.jpeg'
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
    #results_dict = show_image(IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT), color_mode=-1)

    # create random image
    random_image = np.random.randint(255, size=(500,500), dtype=np.uint8)

    # show iamge
    cv2.imshow('Random Image', random_image)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    # create random image
    dark_image = np.random.randint(50, size=(500,500), dtype=np.uint8)

    # show iamge
    cv2.imshow('Dark Image', dark_image)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    # create random image
    light_image = np.random.randint(low=205, high=255, size=(500,500), dtype=np.uint8)

    cv2.imshow('Light Image', light_image)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()
  

    # image histogram
    plt.hist(x=random_image.ravel(), bins=256, range=[0, 256], label='original')
    plt.hist(x=dark_image.ravel(), bins=256, range=[0, 256], label='dark')
    plt.hist(x=light_image.ravel(), bins=256, range=[0, 256], label='light')
    plt.legend()
    plt.show()

    ## histogram equalization

    # equalize images
    equalize_rand_img = cv2.equalizeHist(random_image)
    cv2.imshow('Random Image (equalized)', equalize_rand_img)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    equalize_light_img = cv2.equalizeHist(light_image)
    cv2.imshow('Light Image (equalized)', equalize_light_img)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    equalize_dark_img = cv2.equalizeHist(dark_image)
    cv2.imshow('Dark Image (equalized)', equalize_dark_img)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    # image histograms
    plt.hist(x=equalize_rand_img.ravel(), bins=256, range=[0, 256], label='original')
    plt.hist(x=equalize_light_img.ravel(), bins=256, range=[0, 256], label='light')
    plt.hist(x=equalize_dark_img.ravel(), bins=256, range=[0, 256], label='dark')
    plt.legend()
    plt.show()

    ## equalizing mixed light and dark images

    # concatonate light and dark images
    light_dark_image = np.concatenate((light_image, dark_image), axis=1)
    cv2.imshow('Light/Dark Image', light_dark_image)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    equalize_light_dark_img = cv2.equalizeHist(light_dark_image)
    cv2.imshow('Light/Dark Image (equalized)', equalize_light_dark_img)
    cv2.waitKey(1000) 
    cv2.destroyAllWindows()

    # histogram
    plt.hist(x=light_dark_image.ravel(), bins=256, range=[0, 256], label='original')
    plt.hist(x=equalize_light_dark_img.ravel(), bins=256, range=[0, 256], label='equalized')
    plt.legend()
    plt.show()


##################################################################################################
## END
##################################################################################################

if __name__ == '__main__':
    main()