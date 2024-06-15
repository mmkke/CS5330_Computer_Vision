"""
Lecture4 - Filtering

Michael Massone
5/29/2024
CS5330
Lecture4
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
IMAGE_FILENAME = 'flowers'
FILE_EXT = '.jpg'
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

sobelV = (1/8) * np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]]
                               )

sobelH = (1/8) * np.array([[-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]]
                               )

#gaussian_blur = np.array()

noise_ratios = [0.01, 0.1, 0.5]

##################################################################################################
## Main
##################################################################################################
def main():

    
    # read and show grayscale image
    grscl_image_name, grscl_img = show_image((IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT)), 
                                                     color_mode=0,
                                                     image_name= IMAGE_FILENAME +'_grayscale',
                                                     show=False
                                                     )
    

    img_sobel_horizontal = apply_filter(grscl_img, filter=sobelH)

    img_Sobel_vertical = apply_filter(grscl_img, filter=sobelV)
    
   
    # create 3x3 subplot for images
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

    axs[0].imshow(grscl_img, cmap='gray')
    axs[0].set_title(f'Original')
    axs[0].axis('off')

    axs[1].imshow(img_sobel_horizontal, cmap='gray')
    axs[1].set_title(f'Sobel (Horizontal)')
    axs[1].axis('off')

    axs[2].imshow(img_Sobel_vertical, cmap='gray')
    axs[2].set_title(f'Sobel (Vertical) ')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig('images/lecture4_edge_detection_results.png')
    plt.show()
    

##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()