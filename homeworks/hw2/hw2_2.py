"""
Visualizing Blurring in 3D: 20 points

Michael Massone
6/5/2024
CS5330
HW 2, Part 2

Instructions: 

* Download the dog.jpeg from canvas and convert that to a grey scale
image.
* Run the dog photo through both Sobel filters (x and y) and combine
the results using a smaller cut off value (I used a cut off of 50).
* Run the dog photo through both Sobel filters (x and y) and combine
the results using a larger cut off value (I used a cut off of 150).
* Run the dog photo using the built in cv2.Canny(imgage, 100, 100). (I
used a min and max both of 100).
* Run a 5x5 Gaussian Blur on the dog photo and run the sobel filter and
canny filter on the blurred image.
* Use a 2x4 grid to show all your results.
* Answer the following questions at the end of your code by leaving a
comment:
    - What did you notice when you went from a lower threshold value
to a higher one?
    - What did you notice before and after applying a Gaussian Blur
to the image?
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
IMAGE_FILENAME_1 = 'dog'
FILE_EXT = '.jpeg'
IMAGE_PATH = BASE_DIR / 'images' # path to images

print('Current working directory:', BASE_DIR)


##################################################################################################
## Modules
##################################################################################################
# Get the parent directory
parent_dir = Path(__file__).resolve().parent.parent.parent

# Add the parent directory to sys.path
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
import cv_utils as cv

##################################################################################################
## Functions
##################################################################################################
def combine_sobels(img1, img2, threshold):

    # create new matrix
    img_combined = np.zeros_like(img1)
    # get magnitude of hte gradient 
    g = np.sqrt(img1**2 + img2**2)
    # normalize the gradient and scale 0->255
    img_combined = np.uint8(g / np.max(g) * 255)
    # threshold and binarize image (if greater than threshold set to 255, otherwise 0)
    img_combined = np.where(img_combined > threshold, 255, 0).astype(np.uint8)
    print(img_combined)

    return img_combined

##################################################################################################
## Kernels
##################################################################################################

sobel_x = (1/9) * np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]
                           ])

sobel_y = (1/9) * np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]
                           ])

##################################################################################################
## Main
##################################################################################################
def main():

    imgs = []

    ## Read in Images
    # img1
    img1_name, img1 = cv.show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0
                                 )
    
    #img1 = cv.histEqualize(img1)

    ## Apply Gaussian in Frequency Domain

    # Create Guassian Filter Mask
    height, width = img1.shape
    gauss_filter_mask = cv.gaussian_filter(height, width, std_dev=30)

    # Apply Fourier Transform and Filter in Frequency Domain
    img1_gaussian = cv.apply_fft_filter(img1, gauss_filter_mask)
    
    # Normalize values to 0->255
    img1_gaussian = cv.normalize_image(img1_gaussian)
    imgs.append(('img1_gaussian', img1_gaussian))

    ## Apply Sobel Kernel in Spatial Domain

    # Apply filters to non-smoothed img
    img1_sobelH = cv.apply_filter(img1, sobel_x)
    img1_sobelV = cv.apply_filter(img1, sobel_y)

    # Blend images
    # threshold = 50
    img1_sobel_50 = combine_sobels(img1_sobelH, img1_sobelV, threshold=100)
    imgs.append(('img1_sobel_50', img1_sobel_50))
    #thresold = 150
    img1_sobel_150 = combine_sobels(img1_sobelH, img1_sobelV, threshold=150)
    imgs.append(('img1_sobel_150', img1_sobel_150))



    # Apply filters to smoothed img
    img1_blur_sobelH = cv.apply_filter(img1_gaussian, sobel_x)
    img1_blur_sobelV = cv.apply_filter(img1_gaussian, sobel_y)

    # equalize
    #img1_blur_sobelH = cv.histEqualize(img1_blur_sobelH)
    #img1_blur_sobelV = cv.histEqualize(img1_blur_sobelV)

    # Blend Images
    # threshold = 50
    img1_blur_sobel_50 = combine_sobels(img1_blur_sobelH, img1_blur_sobelV, threshold=100)
    #img1_blur_sobel_50 = cv.histEqualize(img1_blur_sobel_50)
    imgs.append(('img1_blur_sobel_50', img1_blur_sobel_50))
    # threshold = 150
    img1_blur_sobel_150 = combine_sobels(img1_blur_sobelH, img1_blur_sobelV, threshold=150)
    #img1_blur_sobel_150 = cv.histEqualize(img1_blur_sobel_150)
    imgs.append(('img1_blur_sobel_150', img1_blur_sobel_150))


    # Generate Canny Images
    img1_canny = cv2.Canny(img1, 100, 100)
    imgs.append(('img1_canny', img1_canny))
    img1_blur_canny = cv2.Canny(img1_gaussian, 100, 100)
    imgs.append(('img1_blur_canny', img1_blur_canny))


    # Save images
    for img in imgs:
        pil_img = Image.fromarray(img[1])
        img_name = f'{img[0]}.jpeg'
        pil_img.save(IMAGE_PATH / img_name) 
    
    
    ## Show Images

    ## Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Plot non-smothed images 
    axes[0][0].imshow(img1, cmap='gray')
    axes[0][0].axis('off')
    axes[0][0].set_title('Original Image')

    axes[0][1].imshow(img1_sobel_50, cmap='gray')
    axes[0][1].axis('off')
    axes[0][1].set_title('Sobel (threshold = 100)')

    axes[0][2].imshow(img1_sobel_150, cmap='gray')
    axes[0][2].axis('off')
    axes[0][2].set_title('Sobel (threshold = 150)')

    axes[0][3].imshow(img1_canny, cmap='gray')
    axes[0][3].axis('off')
    axes[0][3].set_title('Canny')

    # Plot smoothed images
    axes[1][0].imshow(img1_gaussian, cmap='gray')
    axes[1][0].axis('off')
    axes[1][0].set_title('Smoothed Imaged')

    axes[1][1].imshow(img1_blur_sobel_50, cmap='gray')
    axes[1][1].axis('off')
    axes[1][1].set_title('Sobel (threshold = 100)')

    axes[1][2].imshow(img1_blur_sobel_150, cmap='gray')
    axes[1][2].axis('off')
    axes[1][2].set_title('Sobel (threshold = 150)')

    axes[1][3].imshow(img1_blur_canny, cmap='gray')
    axes[1][3].axis('off')
    axes[1][3].set_title('Canny')


    plt.tight_layout()
    plt.savefig('images/results_pt2.png')
    plt.show()

    '''
    Going from a lower threshold to a higher thereshold on the Soble filter removed some the extra edges that were 
    not helpful in depectiing the images boundary.
    
    Applying a gaussian blur before running the edge detection algorithms improved beformance. By reducing noise 
    and the sharpness of the image the algorithms were better able idenitfy only the most significant edges that 
    ultimately best represented the image. For example, is the unsmoother image, the texture of the dogs fur produced 
    many edges that did not contribute the a recognizable image. By reducing the contrast in these regions, smoothing 
    the image reduces those edges making hte outline of the dog easier to identify.   '''


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()