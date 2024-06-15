# Michael Massone
# 5/8/2024

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import cv2

img = cv2.imread('images/testimage1.png', 0)
cv2.imshow('image', img)

cv2.waitKey(5000) 
cv2.destroyAllWindows() 

#plt.imshow(img, cmap='gray')
#plt.axis('off')  
#plt.show()

