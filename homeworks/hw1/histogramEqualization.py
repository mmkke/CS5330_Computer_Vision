"""
Creation of a histogram equalization function and comparison of results to cv2.equalizeHist.

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
IMAGE_FILENAME = 'test_image'
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
        # create openCV image obj
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
    
    # read image
    image_name, img = show_image((IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT)))

    # equalize image
    equalized_img = histEqualize(img)
    equalized_img_name = 'dark_image_equalized'
    equalized_img_path = equalized_img_name + '.jpg'

    # save image a .jpg using pil
    pil_img = Image.fromarray(equalized_img)
    pil_img.save(IMAGE_PATH / equalized_img_path)

    # get cv2.equalizeHist() image
    cv2_equalized_img = cv2.equalizeHist(img)
    cv2_equalized_img_name = 'dark_image_equalized_cv2'
    cv2_equalized_img_path = cv2_equalized_img_name + '.jpg'

    # save image a .jpg using pil
    pil_img = Image.fromarray(cv2_equalized_img)
    pil_img.save(IMAGE_PATH / cv2_equalized_img_path)

    ## Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 5))

    # plot each image 
    axes[0][0].imshow(img, cmap='gray')
    axes[0][0].axis('off')
    axes[0][0].set_title(image_name)

    axes[1][0].imshow(equalized_img, cmap='gray')
    axes[1][0].axis('off')
    axes[1][0].set_title(equalized_img_name)

    axes[2][0].imshow(cv2_equalized_img, cmap='gray')
    axes[2][0].axis('off')
    axes[2][0].set_title(cv2_equalized_img_name)

    # plot corresponding histograms
    axes[0][1].hist(img.ravel(), bins=256, range=[0, 256])
    axes[1][1].hist(equalized_img.ravel(), bins=256, range=[0, 256])
    axes[2][1].hist(cv2_equalized_img.ravel(), bins=256, range=[0, 256])

    plt.tight_layout()
    plt.savefig('images/histogram_results.png')
    plt.show()

##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()