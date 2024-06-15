"""
Lecture 4 - Locally adaptive histogram equalization

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
IMAGE_FILENAME = 'Dark_Light'
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
def histEqualize(img):
    ''' 
    Description: Equalizes a grayscale image using it's CDF.

    Inputs:
            img (ndarray): Grayscale iamge as ndarray
             
    Outputs:
            equalized_img (ndarray): equalized image

    '''
    
    # get value counts for numpy array
    values, counts = np.unique(img, return_counts=True)
    n_pixels = img.size
    print('Image array dimensions: ', img.shape)
    print('Number of pixels: \n', n_pixels)

    # use np.cumsum to calculate the cdf
    cdf = np.cumsum(counts)
    print('CDF: \n', cdf)

    # get the min value of the cdf
    cdf_min = np.min(cdf)
    print('CDF min value: \n', cdf_min)

    # use the histogram equaliztion equation
    cdf_norm = ((cdf - cdf_min) / (n_pixels - cdf_min)) * 255
    print('cdf norm: ', cdf_norm)

    # interpolate the values of the euqalized image using the original image and the normalized cdf
    # convert to .uint8 and reshape array to roginal dimensions
    equalized_img =  np.interp(x = img.flatten(), xp = values, fp = cdf_norm).astype(np.uint8).reshape(img.shape)
    print('equalized img: ', equalized_img)

    return equalized_img


##################################################################################################
## Main
##################################################################################################
def main():

    # create dark_light image
    Dark_image = np.random.randint(50, size=(500, 500),dtype=np.uint8)
    Light_image = np.random.randint(low = 205, high = 255, size=(500, 500),dtype=np.uint8)
    Dark_Light = np.concatenate((Dark_image, Light_image), axis=1)

    # save img as .jpg to /images
    Dark_Light_img_name = 'Dark_Light'
    dark_light_img_path = Dark_Light_img_name + '.jpg'
    pil_img = Image.fromarray(Dark_Light)
    pil_img.save(IMAGE_PATH / dark_light_img_path)

    # read/show img
    img_name, img = show_image(IMAGE_PATH / dark_light_img_path, 
                               image_name=Dark_Light_img_name, 
                               color_mode=0, show=True)
    
    # get hieght and width
    height, width = img.shape
    print(height, width)

    # get step sizes for 10 subdivisions
    vertical_step = int(height/2)
    horizontal_step = int(width/5)
    print(vertical_step, horizontal_step)

    # create zeros like array to assign equalized img values
    equal_img = np.zeros_like(img)

    # loop through original image by section and equalize, assigning new values to zeros like array
    for i in range(0, height, (vertical_step)):
        for j in range(0, width, (horizontal_step)):
            equal_img[i:i+vertical_step, j:j+horizontal_step] = histEqualize(img[i:i+vertical_step, j:j+horizontal_step])
    
    # show result
    plt.imshow(equal_img, cmap='gray')
    plt.show()


    ## cv2 Contrast Limiting AHE

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize = (200,200))
    clahe_img = clahe.apply(img)
    plt.imshow(clahe_img, cmap='gray')
    plt.show()

    


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()