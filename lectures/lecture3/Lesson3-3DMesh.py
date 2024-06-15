#Ryan Bockmon
#5/20/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/dog.jpeg',2)#reads in an image as grey scale
cv2.imshow('image',img)



height,width=img.shape#returns the height and width of the image
x = np.linspace(0,width,width,dtype = int)#creats an array from 0->width
y = np.linspace(0,height,height,dtype = int)#creats an array from 0->height
X, Y = np.meshgrid(x,y)#creats a 2D meshgrid 


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')#creats a subplot that can plot in 3D
surf = ax.plot_surface(X,Y,img,cmap=plt.cm.gray)#plots the mesh grid with the image values
#cmap = color map
plt.show()
