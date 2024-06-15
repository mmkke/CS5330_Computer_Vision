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
IMAGE_FILENAME_1 = 'blocks1'
FILE_EXT = '.jpg'
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


##################################################################################################
## Main
##################################################################################################
def main():

    print(str(IMAGE_PATH / IMAGE_FILENAME_1))
    img1 = cv2.imread('/Users/mikey/roux_class_files/CS5330/lectures/lecture6/images/blocks1.jpg')

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap='gray')
    plt.show()

    plt.hist(gray.ravel(), 256, [0, 256])
    plt.show()

    ret, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    plt.imshow(binary, cmap='gray')
    plt.show()

    inv = 255 - binary
    plt.imshow(inv, cmap='gray')
    plt.show()

    kernel = np.ones((7,7), np.uint8)

    img_dilation = cv2.dilate(inv, kernel, iterations=1)
    plt.imshow(img_dilation, cmap='gray')
    plt.show()

    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    plt.imshow(img_erosion, cmap='gray')
    plt.show()

    contours, hierarchy = cv2.findContours(img_erosion, 
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
    

    # -1 = draw all contours
    cv2.drawContours(img1, contours, -1, (0,255,0), 3)

    plt.imshow(img1)
    plt.show()

##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()