'''
CS5330 Lecture2
5/15/2024
Michael Massone
'''
################################################################################################
## Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################################################
## Functions


################################################################################################
## Main

def main():

    # read in original image
    img1 = cv2.imread("images/testimage1.png")
    cv2.imshow("image1", img1)
    cv2.waitKey(1000)
    height, width, channels = img1.shape
    print("img1")
    print('height: ', height, '\nwidth: ', width, '\nchannel: ', channels)
    print(img1)

    # read image in as greyscale
    img2 = cv2.imread("images/testimage1.png", 0)
    cv2.imshow("image1_greyscale", img2)
    cv2.waitKey(1000)
    height, width = img2.shape
    print("img2")
    print('height: ', height, '\nwidth: ', width)
    print(img2)

    # creat inverted image
    inverter = np.full((width, height), 255, dtype=np.uint8)
    img2_inverted = inverter - img2
    cv2.imshow("image2_inverted", img2_inverted)
    cv2.waitKey(1000)

    # create your own image in greyscale 
    img3 = np.random.randint(255, size=(500, 500), dtype=np.uint8)
    cv2.imshow("image", img3)
    cv2.waitKey(1000)
    height, width = img3.shape
    print("img3")
    print('height: ', height, '\nwidth: ', width)
    print(img3)

    # create your own image in RGB 
    img4 = np.random.randint(255, size=(500, 500, 3), dtype=np.uint8)
    cv2.imshow("image", img4)
    cv2.waitKey(1000)
    height, width, channels = img4.shape
    print("img4")
    print('height: ', height, '\nwidth: ', width, '\nchannel: ', channels)
    print(img4)

    # save images
    cv2.imwrite("images/img2_inverted.png", img2_inverted)
    cv2.imwrite("images/img3.png", img3)
    cv2.imwrite("images/img4.png", img4)

    file_name = "images/dark_image.jpg"
    dark_img = cv2.imread(file_name, 0)
    cv2.imshow("dark_image", dark_img)
    cv2.waitKey(2000)
    plt.figure("Histogram of Image")
    plt.hist(dark_img.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram of Image')
    plt.show()
  
    equ = cv2.equalizeHist(dark_img)
    cv2.imshow("dark_image_equ", equ)
    cv2.waitKey(2000)
    plt.figure("Histogram of Equalized Image")
    plt.hist(equ.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram of Equalized Image')
    plt.show()


    cv2.destroyAllWindows()



################################################################################################
## END

if __name__ == "__main__":
    main()