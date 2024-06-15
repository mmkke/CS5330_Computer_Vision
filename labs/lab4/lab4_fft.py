"""
Marlyn Einstein - low/high pass filtered images. 

Michael Massone
6/5/2024
CS5330
Lab 4

References:
1) Class Textbook
2) https://www.youtube.com/watch?v=OOu5KP3Gvx0&list=PL2zRqk16wsdorCSZ5GWZQr1EMWXs2TDeu&index=11
3) https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
4) https://www.geeksforgeeks.org/how-to-find-the-fourier-transform-of-an-image-using-opencv-python/
5) https://numpy.org/doc/stable/reference/routines.fft.html
6) https://www.geeksforgeeks.org/normalize-an-image-in-opencv-python/
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


    ## Create Guassian Filter Mask
    size, _ = img1.shape
    gauss_filter_mask = gaussian_filter(size, std_dev=10)

    ## Apply Fourier Transform and Filter in Frequency Domain
    img1_low = apply_fft_filter(img1, gauss_filter_mask)

   
    ## High Pass Filter in Frequency Domain
    cutoff = 30
    high_filter_mask = high_pass_filter(size, cutoff)
    img2_high = apply_fft_filter(img2, high_filter_mask)
  

    ## Normalize values to 0->255

    img1_low = normalize_image(img1_low)
    img2_high = normalize_image(img2_high)

    ## Equalize
    #img1_low = histEqualize(img1_low)
    #img2_high = histEqualize(img2_high)


    ## Combine Images

    # set alhpha for a 50/50 blend of the two images
    alpha = 0.5
    combined_img = (img1_low * alpha + img2_high * (1 - alpha)).astype(np.uint8) 

    ## Equalize
    #combined_img = histEqualize(combined_img)   

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
    print('***************************************')    
    print("img1:")
    print('***************************************')  
    print(img1_low)
    print('***************************************')
    print("img2:")
    print('***************************************')
    print(img2_high)
    print('***************************************')
    print("combined:")
    print('***************************************')
    print(combined_img)

    ## Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 5))

    # plot each image 
    axes[0][0].imshow(img1_low, cmap='gray')
    #axes[0][0].imshow(img1, cmap='gray')
    axes[0][0].axis('off')
    axes[0][0].set_title('Low Pass (fft)')

    axes[1][0].imshow(img2_high, cmap='gray')
    #axes[1][0].imshow(img2, cmap='gray')
    axes[1][0].axis('off')
    axes[1][0].set_title('High Pass (fft)')

    axes[2][0].imshow(combined_img, cmap='gray')
    axes[2][0].axis('off')
    axes[2][0].set_title('Combined Image (fft)')

    # plot corresponding histograms
    axes[0][1].hist(img1_low.ravel(), bins=256, range=[0, 256])
    #axes[0][1].hist(img1.ravel(), bins=256, range=[0, 256])
    axes[1][1].hist(img2_high.ravel(), bins=256, range=[0, 256])
    #axes[1][1].hist(img2.ravel(), bins=256, range=[0, 256])
    axes[2][1].hist(combined_img.ravel(), bins=256, range=[0, 256])

    plt.tight_layout()
    plt.savefig('images/results_fft.png')
    plt.show()


##################################################################################################
## END
##################################################################################################
if __name__ == '__main__':
    main()