"""
Lab3 - Neighborhood filters.

Michael Massone
5/29/2024
CS5330
Lab3
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
IMAGE_FILENAME = 'dog'
FILE_EXT = '.jpeg'
IMAGE_PATH = BASE_DIR / 'images' # path to images

print('Current working directory:', BASE_DIR)


##################################################################################################
## Functions
##################################################################################################
def show_image(image_path: Path, color_mode=0, image_name=IMAGE_FILENAME, show=False):
    ''' 
    Description: This function displays an image using openCV.

        https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html

    Inputs:
            image_path (Path): Path to image.
             color_mode(int)->0: openCV image colormode:
                                                     0 = grayscale (default)
                                                     1 = RGB
                                                    -1 = unchanged for original image
            image_name (str): Name for image, default is file name of image. 
            show(bool)->False: If True will display image using matplotlib.

    Outputs:
            image_name (str): Image name 
                   img (ndarray): cv2 image

    '''

    try:
        # Create openCV image obj
        img = cv2.imread(str(image_path), color_mode)

        # Dheck img create successfully
        if img is None:
            raise FileNotFoundError(f"No image found at {str(image_path)}.")
        
        
        # convert color channel order
        if color_mode != 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 (BGR) -> matplotlib (RGB)

        # display image
        if show: 
            if color_mode == 0:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(image_name)
            plt.show()

        return  image_name, img
         
    
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return None

##################################################################################################
def add_random_noise(img, noise_ratio):
    ''' 
    Description: Adds random salt and pepper noise to image.

    Inputs:
            img (ndarray): Grayscale image as ndarray
            noise_ratio(float): ratio of noise.
             
    Outputs:
            img (ndarray): noised image image

    '''
    # set max x and y values based on img dimensions
    x_max = img.shape[0]
    y_max = img.shape[1]
    
    # assign salt and pepper noise to random pixels
    for i in range(int(img.size * noise_ratio)):
        rand_x = np.random.randint(0, x_max)
        rand_y = np.random.randint(0, y_max)
        rand_color = np.random.choice([0, 255])
        img[rand_x, rand_y] = rand_color

    return img

##################################################################################################
def apply_filter(img, filter):
    ''' 
    Description: Applys a filter to an image using the zeros padding method.

    Inputs:
            img (ndarray): Grayscale image as ndarray
            filter(ndarray): Filter kernel.
             
    Outputs:
            filtered_img (ndarray): noised image image
    '''

    filtered_img = np.zeros_like(img)
    filter_size = len(filter)
    pad_width = filter_size // 2

    # create padded image
    padded_img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

    # apply filter
    for i in range(padded_img.shape[0]-(filter_size-1)):
        for j in range(padded_img.shape[1]-(filter_size-1)):
            neighborhood_matrix = padded_img[i:i+filter_size, j:j+filter_size]
            #print('loc: ', i, j)
            #print('dimensions: ', padded_img.shape)
            #print(neighborhood_matrix)
            filtered_img[i, j] = np.sum(neighborhood_matrix * filter)

    return filtered_img
##################################################################################################
## Parameters
##################################################################################################

filter_3x3 = (1/9) * np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]]
                               )

filter_5x5 = (1/25) * np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 0, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]]
                                )

sobel = (1/9) * np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]]
                               )

noise_ratios = [0.01, 0.1, 0.5]

##################################################################################################
## Main
##################################################################################################
def main():

    
    # read and show grayscale image
    grscl_image_name, grscl_img = show_image((IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT)), 
                                                     color_mode=0,
                                                     image_name= IMAGE_FILENAME +'_grayscale'
                                                     )
    # create 3x3 subplot for images
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    for i, noise_ratio in enumerate(noise_ratios):

        img_unfiltered = add_random_noise(grscl_img, noise_ratio)

        img_filtered_3x3 = apply_filter(img_unfiltered, filter=filter_3x3)

        img_filtered_5x5 = apply_filter(img_unfiltered, filter=filter_5x5)

        axs[i][0].imshow(img_unfiltered, cmap='gray')
        axs[i][0].set_title(f'Unfiltered \nNoise Ratio: {noise_ratio}')
        axs[i][0].axis('off')

        axs[i][1].imshow(img_filtered_3x3, cmap='gray')
        axs[i][1].set_title(f'Filtered (3x3) \nNoise Ratio: {noise_ratio}')
        axs[i][1].axis('off')

        axs[i][2].imshow(img_filtered_5x5, cmap='gray')
        axs[i][2].set_title(f'Filtered (5x5) \nNoise Ratio: {noise_ratio}')
        axs[i][2].axis('off')

    plt.tight_layout()
    plt.savefig('images/lab3_results.png')
    plt.show()
    

##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()