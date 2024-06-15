"""
Lecture 3 Module - 3D image projection

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
from mpl_toolkits.mplot3d import Axes3D


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

##################################################################################################
## Main
##################################################################################################
def main():

    #results_dict = show_image(IMAGE_PATH / (IMAGE_FILENAME + FILE_EXT), color_mode=0)

    img = cv2.imread('images/dog.jpeg', 0)

    height, width = img.shape

    # get pixel locations (matrix dimensions)
    x = np.linspace(start=0, stop=width, num=width, dtype=int)
    y = np.linspace(start=0, stop=height, num=height, dtype=int)

    # create 2d planes (mesh)
    X,Y = np.meshgrid(x,y)
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    surface = ax.plot_surface(X,Y,img, cmap=plt.cm.gray)
    plt.show()
  


##################################################################################################
## END
##################################################################################################

if __name__ == '__main__':
    main()