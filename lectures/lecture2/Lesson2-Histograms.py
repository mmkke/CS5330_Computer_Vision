#Ryan Bockmon
#5/20/2024
import cv2
import numpy as np
import matplotlib.pyplot as plt


Random_image = np.random.randint(255, size=(500, 500),dtype=np.uint8)
Dark_image = np.random.randint(50, size=(500, 500),dtype=np.uint8)
Light_image = np.random.randint(low = 205, high = 255, size=(500, 500),dtype=np.uint8)


#cv2.imshow('image1',Random_image)
#cv2.imshow('image2',Dark_image)
#cv2.imshow('image3',Light_image)

"""
plt.hist(Random_image.ravel(),256,[0,256],histtype=u'step')
plt.hist(Dark_image.ravel(),256,[0,256],histtype=u'step')
plt.hist(Light_image.ravel(),256,[0,256],histtype=u'step')
plt.show()
"""

#histogram equalization
"""
equalize_img1 = cv2.equalizeHist(Random_image)
equalize_img2 = cv2.equalizeHist(Dark_image)
equalize1_img3 = cv2.equalizeHist(Light_image)


fig, ax = plt.subplots(2, 1)#allows for multiple plots to be shown
ax[0].hist(Dark_image.ravel(),256,[0,256],histtype=u'step')
ax[1].hist(equalize_img2.ravel(),256,[0,256],histtype=u'step')
plt.show()

cv2.imshow('Non-Equalize',Dark_image)
cv2.imshow('Equalized',equalize_img2)


Dark_Light = np.concatenate((Dark_image, Light_image), axis=1) 
cv2.imshow('Non-Equalize',Dark_Light)
equalize = cv2.equalizeHist(Dark_Light)
cv2.imshow('Equalized',equalize)


fig, ax = plt.subplots(2, 1)
ax[0].hist(Dark_Light.ravel(),256,[0,256],histtype=u'step')
ax[1].hist(equalize.ravel(),256,[0,256],histtype=u'step')
plt.show()
"""

img = cv2.imread('images/flowers.jpg')
#cv2.imshow('image1',img)
#plt.hist(img.ravel(),256,[0,256],histtype=u'step')#only works for grey scale
#hist = cv2.calcHist([img],[0],None,[256],[0,256])#returns a 1d list of histgram value of channel 0 -> blue

"""
colors = ['b','g','r']
for i in range(3):#loops for each number of colors/channels
    hist = cv2.calcHist([img],[i],None,[256],[0,256])#returns a 1d list of histgram value of channel 0 -> blue
    plt.plot(hist, color = colors[i])
plt.show()
"""


#equalize = cv2.equalizeHist(img)#does not work with color images
equalize_b = cv2.equalizeHist(img[:,:,0])#Equalizes each individual channel
equalize_g = cv2.equalizeHist(img[:,:,1])
equalize_r = cv2.equalizeHist(img[:,:,2])

#combines each equalized channel
equalized_img = np.stack((equalize_b,equalize_g,equalize_r), axis=2)
colors = ['b','g','r']
for i in range(3):#loops for each number of colors/channels
    hist = cv2.calcHist([equalized_img],[i],None,[256],[0,256])#returns a 1d list of histgram value of channel 0 -> blue
    plt.plot(hist, color = colors[i])
plt.show()

cv2.imshow('Equalized',equalized_img)


