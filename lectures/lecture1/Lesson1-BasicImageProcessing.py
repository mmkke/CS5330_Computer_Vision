#Ryan Bockmon
#5/13/2024
import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('images/testimage1.png')#Opens the image in color
img2 = cv2.imread('images/testimage1.png', cv2.IMREAD_GRAYSCALE)#Opens the image in greyscale (2)

#print(img3)#prints a summary of the pixle information
"""
print(img1)
print(img1[100,100])#prints a single pixle value for the color image [r,g,b]
print(img2[100,100])#prints a single pixle value
#cv2.imshow('image',img1)

print(img1.shape)#returns the resolution of an image with # channels
print(img2.shape)#returns the resolution of an image
height,width,channels=img1.shape
print(height)
"""

"""
#-----create your own image-------
#creates a random grey scale with vaules between 0-255 image size 500,500
img = np.random.randint(255, size=(500, 500),dtype=np.uint8)
print(img)
cv2.imshow('image',img)
cv2.imwrite('Random_Image.jpg', img)#saves the image as "Random_Image.jpg"
"""

"""
#Active Learning 2
#Inverse Grey Scale
height, width = img3.shape
#creates a new blank image the same size as orginal
Inverse_Filter = np.full((height,width),255,dtype=np.uint8)
Inverse_Image = Inverse_Filter - img3
cv2.imshow('image',Inverse_Image)
"""

#histogram examples
"""
#print(img3.ravel())#changes the image into a 1d array of values
#cv2.imshow('image',img3)
#plt.hist(img3.ravel(),256,[0,256])
#plt.show()
#plt.hist(1d array of values, number of bins, range of values)

Random_image = np.random.randint(255, size=(500, 500),dtype=np.uint8)
Dark_image = np.random.randint(40, size=(500, 500),dtype=np.uint8)
Light_image = np.random.randint(low = 200, high = 255, size=(500, 500),dtype=np.uint8)

cv2.imshow('image1',Random_image)
#cv2.imshow('image1',Dark_image)
#cv2.imshow('image2',Light_image)
plt.hist(Random_image.ravel(),256,[0,256],histtype=u'step')
plt.hist(Dark_image.ravel(),256,[0,256],histtype=u'step')
plt.hist(Light_image.ravel(),256,[0,256],histtype=u'step')
plt.show()
"""

"""
#histogram equalization
img4 = cv2.imread('images/dark_image.jpg',cv2.IMREAD_GRAYSCALE)#Opens the image in color
#cv2.imshow('image1',img4)
#plt.hist(img4.ravel(),256,[0,256],histtype=u'step')
#plt.show()

equ = cv2.equalizeHist(img4)
cv2.imshow('image1',img4)
cv2.imshow('image2',equ)
plt.hist(img4.ravel(),256,[0,256],histtype=u'step')
plt.hist(equ.ravel(),256,[0,256],histtype=u'step')
plt.show()
"""
