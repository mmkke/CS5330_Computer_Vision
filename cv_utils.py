"""
Computer Vision Utilities

Funtions:
            1) show_image()
            2) resize_image()
            3) gaussian_filter()
            4) high_pass_filter()
            5) cooley_tukey()
            6) apply_fft_filter()
            7) histEqualize()
            8) normalize_image()
            9) apply_filter()
            10) add_random_noise()
            11) combine_sobels()
            
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
## 1
##################################################################################################
def show_image(image_path: Path, image_name, color_mode=0, bgr=False, show=False):
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
        if color_mode != 0 and bgr == False:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 (BGR) -> matplotlib (RGB)
            print("BGR -> RGB")

        # display image
        if show: 
            if color_mode == 0:
                plt.imshow(img, cmap='gray')
            elif bgr == True:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
## 2
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
## 3 
##################################################################################################
def gaussian_filter(height, width, std_dev) -> np.array:
    '''
    Description: 

        Creates a Gaussian filter.
    
    Parameters:

        height (int): Image pixel height.
        weight (int): Image pixel width
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
## 4
##################################################################################################
def high_pass_filter(size, cutoff) -> np.array:
    '''
    Description: 
    
        Performs a gaussian high pass filter operation.
    
    Parameters:

        img_low (np.array): The a low pass filtered copy of the original image.
        img_original (np.array): The unfiltered original version of the image.
    
    Returns:

        img_high (np.array): A high pass filtered image. 
    '''
    
    # define a distribution
    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(x, y)
    
    # get the distance from the center
    dist = np.sqrt(x**2 + y**2)
    
    # calculate a high-pass filter mask
    filter = 1 - np.exp(-(dist**2 / (2 * (cutoff**2))))
    # normalize
    filter_normed = filter/np.sum(filter)

    plt.imshow(filter_normed, cmap='gray')
    plt.colorbar()
    plt.title('High Pass Filter')
    plt.show()
    return filter_normed
##################################################################################################
## 5 
##################################################################################################
def cooley_tukey(img, inverse=False) -> np.array:
    '''
    Description:
    
    Parameters:
    
    Returns:
    '''

    # define fft function
    def fft(arr, inverse):
        
        # get length of array
        N = len(arr)

        # check if array lenvth = power of 2, if not pad with zeros until true
        while np.log2(N) % 1 != 0:
            arr = np.append(arr, 0)
            N = len(arr)

        # base case
        if N <= 1:
            return arr
        
        # divide array into even and odd
        even = fft(arr[0::2], inverse)
        odd = fft(arr[1::2], inverse)

        # get the exponent value from fft equation
        T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]

        # recombine the results
        result = [even[k] + T[k] for k in range(N // 2)] + \
                [even[k] - T[k] for k in range(N // 2)]

        # extra step if going in opposite direction
        if inverse:
            return [val / N for val in result]
        
        else:
            return result
        
    # Apply 1D FFT to each row
    fft_rows = np.array([fft(row, inverse) for row in img])
    
    # Apply 1D FFT to each column of the result
    fft_2d = np.array([fft(col, inverse) for col in fft_rows.T]).T
    
    return fft_2d
##################################################################################################
## 6
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
## 7
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
## 8
##################################################################################################
def normalize_image(image, alpha=0, beta=255):
    # Normalize image to range [0, 1]
    norm_image = cv2.normalize(image, None, alpha, beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image.astype(np.uint8)
##################################################################################################
## 9
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
## 10
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
## 10
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
## 11
##################################################################################################

def combine_sobels(img1, img2, threshold):

    # create new matrix
    img_combined = np.zeros_like(img1)

    # get magnitude of gradient
    #g = np.sqrt((img1 / 2)**2 + (img2 / 2)**2)

    # threshold
    #img_combined = np.where(g > threshold, 255, 0).astype(np.uint8)
    g = np.sqrt(img1**2 + img2**2)
    img_combined = np.uint8(g / np.max(g) * 255)
    img_combined = np.where(img_combined > threshold, 255, 0).astype(np.uint8)

    return img_combined