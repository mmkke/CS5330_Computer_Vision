"""
Visualizing Blurring in 3D: 20 points

Michael Massone
6/5/2024
CS5330
HW 2, Part 1

Instructions: 

* Download the dog.jpeg from canvas and convert it to a grey scale
image.
* Plot the original dog photo in 3D like we did in class.
* Run a 5x5 Gaussian Blur on the dog photo and plot the blurred image
in 3D.
* Run a 11x11 Gaussian Blur on the dog photo and plot the blurred
image in 3D.
* Display Your 3 (original, 5x5, 11x11) graphs at the same time to make
it easier for the TAs to grade.
* Answer the following question by adding it to the end of your code as
a comment: What do you notice about the 3D graphs as the filter size
increases?
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
from matplotlib.gridspec import GridSpec
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
def gaussian_filter(height, width, std_dev) -> np.array:
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
    x = np.linspace(-(width // 2), width // 2, width)
    y = np.linspace(-(height // 2), height // 2, height)
    x, y = np.meshgrid(x, y)

    # calcuate gaussian
    dist = np.sqrt(x**2 + y**2)
    filter = np.exp(-(dist**2 / (2.0 * std_dev**2)))
    # normalize
    filter_normed = filter/ np.sum(filter)

    print('gaussian filter')
    print(filter_normed)
    plt.imshow(filter_normed, cmap='gray')
    plt.colorbar()
    plt.title('Gaussian Filter')
    plt.show()
    return filter_normed


##################################################################################################
def apply_fft_filter(img, filter_mask) -> np.array:
    '''
    Description: Applies the FFT to an image, filters it in the frequency domain, and applies the inverse FFT.
    
    Parameters:
        img (np.array): The input image.
        filter_mask (np.array): The filter mask to apply in the frequency domain.
    
    Returns:
        filtered_img (np.array): The filtered image in the spatial domain.
    '''

    # Compute the FFT of the image
    fft_image = np.fft.fft2(img)
    #fft_image = cooley_tukey(img, inverse=False)

    # Shift to zero
    fft_shifted = np.fft.fftshift(fft_image)

    # Apply the filter mask by multiplying the matricies element-wise
    filtered_fft_shifted = fft_shifted * filter_mask

    # Inverse FFT to get the filtered image back in the spatial domain
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
    filtered_img = np.fft.ifft2(filtered_fft)
    #filtered_img = cooley_tukey(filtered_fft, inverse=True)
    
    # Return the real part of the inverse FFT result, remove imaginary values
    return np.real(filtered_img)

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
def normalize_image(image, alpha=0, beta=255):
    # Normalize image to range [0, 1]
    norm_image = cv2.normalize(image, None, alpha, beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image.astype(np.uint8)

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
def plot_3d(img):

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
## Main
##################################################################################################
def main():

    ## Read in Images
    # img1
    img1_name, img1 = show_image((IMAGE_PATH / (IMAGE_FILENAME_1 + FILE_EXT)),
                                 image_name=IMAGE_FILENAME_1,
                                 color_mode=0
                                 )
    

    ## Apply Gaussian in Frequency Domain

    # Create Guassian Filter Mask
    height, width = img1.shape
    gauss_filter_mask = gaussian_filter(height, width, std_dev=20)

    # Apply Fourier Transform and Filter in Frequency Domain
    img1_gaussian_fft = apply_fft_filter(img1, gauss_filter_mask)

    # Normalize values to 0->255
    img1_gaussian_fft = normalize_image(img1_gaussian_fft)

    # Plot 3d
    #plot_3d(img1_gaussian_fft)

    # save image a .jpg using pil
    pil_img = Image.fromarray(img1_gaussian_fft)
    img_name = 'img1_gaussian_fft.jpeg'
    pil_img.save(IMAGE_PATH / img_name) 


    ## Apply 5x5 Gaussian Kernel in Spatial Domain
    # Create gaussian kernel
    gauss_filter_5x5 = gaussian_filter(height=5, width=5, std_dev=3)

    # Apply filter
    img1_gaussian_5x5 = apply_filter(img1, gauss_filter_5x5)

    # Plot 3d
    #plot_3d(img1_gaussian_5x5)

    # save image a .jpg using pil
    pil_img = Image.fromarray(img1_gaussian_5x5)
    img_name = 'img1_gaussian_5x5.jpeg'
    pil_img.save(IMAGE_PATH / img_name) 

    ## Apply 11x11 Gaussian Kernel in Spatial Domain
    # Ccreate gaussian kernel
    gauss_filter_11x11 = gaussian_filter(height=11, width=11, std_dev=3)

    # Apply filter
    img1_gaussian_11x11 = apply_filter(img1, gauss_filter_11x11)

    # Plot 3d
    #plot_3d(img1_gaussian_11x11)

    # save image a .jpg using pil
    pil_img = Image.fromarray(img1_gaussian_11x11)
    img_name = 'img1_gaussian_11x11.jpeg'
    pil_img.save(IMAGE_PATH / img_name) 
    
    
    ## Show Images

    height, width = img1.shape

    # get pixel locations (matrix dimensions)
    x = np.linspace(start=0, stop=width, num=width, dtype=int)
    y = np.linspace(start=0, stop=height, num=height, dtype=int)

    # create 2d planes (mesh)
    X,Y = np.meshgrid(x,y)

    ## Plot
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(8, 8))

    # plot each image 
    axes[0][0].plot_surface(X, Y, img1, cmap='gray')
    #axes[0][0].axis('off')
    axes[0][0].set_title('Original Image')

    axes[0][1].plot_surface(X, Y, img1_gaussian_fft, cmap='gray')
    #axes[1][0].axis('off')
    axes[0][1].set_title('FFT Gausssian')

    axes[1][0].plot_surface(X, Y, img1_gaussian_5x5, cmap='gray')
    #axes[2][0].axis('off')
    axes[1][0].set_title('5x5 Gaussian')

    axes[1][1].plot_surface(X, Y, img1_gaussian_11x11, cmap='gray')
    #axes[3][0].axis('off')
    axes[1][1].set_title('11x11 Gaussian')


    plt.tight_layout()
    plt.savefig('images/results_pt1.png')
    plt.show()


    '''
    As kernel sizes increases, the resulting image is smoother with fewer local max/mins. This is a 
    result of each pixel being the average of a larger number of local pixel values. When veiwed 
    from the top, the smoothness of the image is less noticebale than the in the 3d representation of 
    the image. This implies that the image can be smoothed without the human eye losing the ability 
    to see the image clearly.
    '''


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()