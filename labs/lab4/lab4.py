"""
Marlyn Einstein - low/high pass filtered images. 

Michael Massone
6/5/2024
CS5330
Lab 4
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
IMAGE_FILENAME_1 = 'scully'
IMAGE_FILENAME_2 = 'mulder'
FILE_EXT = '.jpeg'
IMAGE_PATH = BASE_DIR / 'images' # path to images

print('Current working directory:', BASE_DIR)


##################################################################################################
## Functions
##################################################################################################
def show_image(image_path: Path, image_name, color_mode=0, show=False):
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
def gaussian_filter(size, std_dev) -> np.array:
    '''
    Description: 

        Creates a Gaussian filter.
    
    Parameters:

        size (int): The the number of pixels for the height/weight of the filter.
        std_dev (float): The standard deviation for the gaussian equation. 
    
    Returns:

        filter_normed (np.array): The normalized size x size filter. 
    '''

    # define a distribution
    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(x, y)

    # calcuate gaussian
    dist = np.sqrt(x**2 + y**2)
    filter = np.exp(-(dist**2 / (2.0 * std_dev**2)))
    filter_normed = filter/ np.sum(filter)

    print('gaussian filter')
    print(filter_normed)
    plt.imshow(filter_normed, cmap='gray')
    plt.colorbar()
    plt.show()
    return filter_normed

##################################################################################################
def high_pass_transform(img_low, img_original):
    '''
    Description: 
    
        Performs a high pass filter operation.
    
    Parameters:

        img_low (np.array): The a low pass filtered copy of the original image.
        img_original (np.array): The unfiltered original version of the image.
    
    Returns:

        img_high (np.array): A high pass filtered image. 
    '''
    
    img_high =  (img_original - img_low) + 127
    return img_high



##################################################################################################
def resize_images(img1, img2):
    ''' 
    Description: 
            Resize images to match. 
    Inputs:
            img1 (ndarray):  image as ndarray
            img2 (ndarray):  image as ndarray
             
    Outputs:
            img1 (ndarray): resized image as ndarray
            img2 (ndarray): resized image as ndarray
    '''
    
    if img1.shape > img2.shape:
        size = img2.shape
        img1 = cv2.resize(img1, size)

    elif img2.shape > img1.shape:
        size = img1.shape
        img2 = cv2.resize(img2, size)
    
    assert img1.size == img2.size

    return img1, img2

##################################################################################################
## Main
##################################################################################################
def main():

    ## Read in Images
    # img1
    img1_name, img1 = show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0
                                 )
    
    # img2
    img2_name, img2 = show_image((IMAGE_PATH / (IMAGE_FILENAME_2 + FILE_EXT)), 
                                                     image_name=IMAGE_FILENAME_2,
                                                     color_mode=0
                                                     )

    ## Resize images

    img1, img2 = resize_images(img1, img2)


    ## Low Pass Filter
    filter = gaussian_filter(size=11, std_dev=3)
    img1_low = apply_filter(img1, filter)
    img2_low = apply_filter(img2, filter)


    ## High Pass Filter
    img2_high = high_pass_transform(img2_low, img2) 


    ## Combine Images

    # set alhpha for a 50/50 blend of the two images
    alpha = 0.5
    combined_img = (img1_low * alpha + img2_high * (1 - alpha)).astype(np.uint8)


    ## Show Images

    # save image a .jpg using pil
    pil_img = Image.fromarray(combined_img)
    combined_img_path = 'combined_img.jpeg'
    pil_img.save(IMAGE_PATH / combined_img_path) 

    combined_img_name, combined_img = show_image((IMAGE_PATH / combined_img_path),
                                                 image_name=combined_img_path,
                                                 color_mode=0,
                                                 show=True
                                                )
    
    ## Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 5))

    # plot each image 
    axes[0][0].imshow(img1_low, cmap='gray')
    axes[0][0].axis('off')
    axes[0][0].set_title('Low Pass')

    axes[1][0].imshow(img2_high, cmap='gray')
    axes[1][0].axis('off')
    axes[1][0].set_title('High Pass')

    axes[2][0].imshow(combined_img, cmap='gray')
    axes[2][0].axis('off')
    axes[2][0].set_title('Combined Image')

    # plot corresponding histograms
    axes[0][1].hist(img1_low.ravel(), bins=256, range=[0, 256])
    axes[1][1].hist(img2_high.ravel(), bins=256, range=[0, 256])
    axes[2][1].hist(combined_img.ravel(), bins=256, range=[0, 256])

    plt.tight_layout()
    plt.savefig('images/results.png')
    plt.show()


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()