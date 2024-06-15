"""
Faux thermal vision filter.

Michael Massone
5/29/2024
CS5330
Homework 1
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
IMAGE_FILENAME = 'headshots'
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
def thermalVision(img):
    ''' 
    Description:    Creates a faux thermal vision rendering of a grayscale img by mapping 
                    the grayscale values to an 256 step interpolation from B->G-R.

    Inputs:
            img (ndarray): Grayscale image as ndarray
             
    Outputs:
            thermal_img (ndarray): faux thermal image

    '''
    # get dimensions of image
    height, width = img.shape

    # create new 3d array for color image
    thermal_img = np.zeros([height, width, 3], dtype=np.uint8)

    # define the colors in RGB format
    BLUE = np.array([255, 0, 0])
    GREEN = np.array([127, 255, 127])
    RED = np.array([0, 0, 255])

    # steps to interpolate
    steps = int(256/2)

    # interpolation factors
    interp_factors = np.linspace(0, 1, steps)

    # init lists to hold the interpolated values
    interpolated_colors_B2G = []
    interpolated_colors_G2R = []

    # interpolate between BLUE and GREEN for 0->127
    for factor in interp_factors:
        interpolated_color = (1 - factor) * BLUE + factor * GREEN
        interpolated_colors_B2G.append(interpolated_color)

    # interpolate between GREEN and RED for 128->255
    for factor in interp_factors:
        interpolated_color = (1 - factor) * GREEN + factor * RED
        interpolated_colors_G2R.append(interpolated_color)

    # convert to uint8
    interpolated_colors_B2G = np.array(interpolated_colors_B2G, dtype=np.uint8)
    interpolated_colors_G2R = np.array(interpolated_colors_G2R, dtype=np.uint8)

    # concat array into a single range
    thermal_color_range = np.concatenate((interpolated_colors_B2G, interpolated_colors_G2R), axis=0)
    print(thermal_color_range)

    for i in range(height):
        for j in range(width):
            # print some of the mapping to check
            #if img[i][j] % 50 == 0:
                #print(str(img[i][j]) + " -> " + str(thermal_color_range[img[i][j]]))
            thermal_img[i][j] = thermal_color_range[img[i][j]]

    return thermal_img


##################################################################################################
## Main
##################################################################################################
def main():

    # read and show orignal image
    image_name, img = show_image((IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT)),
                                 color_mode=1
                                 )
    
    # read and show orignal image
    grayscale_image_name, grayscale_img = show_image((IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT)), 
                                                     color_mode=0,
                                                     image_name= IMAGE_FILENAME +'_grayscale'
                                                     )

    # create thermal image
    thermal_img = thermalVision(grayscale_img)
    thermal_img_name = IMAGE_FILENAME + '_thermal'
    thermal_img_path = thermal_img_name + '.jpg'

    # BGR -> RGB
    thermal_img = thermal_img[...,::-1]

    # save image a .jpg using pil
    pil_img = Image.fromarray(thermal_img)
    pil_img.save(IMAGE_PATH / thermal_img_path) 

    # read and show image
    thermal_image_name, thermal_img = show_image((IMAGE_PATH / thermal_img_path), 
                                                 color_mode=1, 
                                                 image_name=thermal_img_name, 
                                                 )

    # plot together
    fig, axs = plt.subplots(3, 1, figsize=(5, 9))

    axs[0].imshow(img)
    axs[0].set_title(image_name)
    axs[0].axis('off')

    axs[1].imshow(grayscale_img, cmap='gray')
    axs[1].set_title(grayscale_image_name)
    axs[1].axis('off')

    axs[2].imshow(thermal_img)
    axs[2].set_title(thermal_image_name)
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig('images/thermal_results.png')
    plt.show()

##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()