#!/usr/bin/env python
# coding: utf-8

# In[1]:


#optimisation required
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
   table = np.array([((i / 255.0) ** gamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def extract_ma(image):
    r,g,b=cv2.split(image)
    comp=255-g
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    histe=clahe.apply(comp)
    adjustImage = adjust_gamma(histe,gamma=3)
    comp = 255-adjustImage
    J =  adjust_gamma(comp,gamma=4)
    J = 255-J
    J = adjust_gamma(J,gamma=4)

    K=np.ones((11,11),np.float32)
    L = cv2.filter2D(J,-1,K)

    ret3,thresh2 = cv2.threshold(L,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    plt.imshow(thresh2)

    kernel2=np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3=np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    return opening


if __name__ == "__main__":
    fundus = cv2.imread("15_right.jpeg")
    bloodvessel = extract_ma(fundus)
    original=bloodvessel

    #this image is for comparison
    contrast = cv2.imread("tree_vessel.png")
    # convert the images to grayscale
   
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    # initialize the figure
    fig = plt.figure("Images")
    images = ("Original", original), ("Contrast", contrast)
    # loop over the images
    for (i, (name, image)) in enumerate(images):
        # show the image
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap = plt.cm.gray)
        plt.axis("off")
    # show the figure
    plt.show()
    
    width = 512
    height = 512
    dim = (width, height)
    
    original=cv2.resize(original,dim,interpolation = cv2.INTER_AREA)
    contrast=cv2.resize(contrast,dim,interpolation = cv2.INTER_AREA)
    err = np.sum((original.astype("float") - contrast.astype("float")) ** 2)
    err /= float(original.shape[0] * original.shape[1])
    m = err
    s = measure.compare_ssim(original, contrast)

    fig = plt.figure("MSE PLOT")
    print(m,s)


    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(original, cmap = plt.cm.gray)
    plt.axis("off")
    
    if(m>0.1 and s<1):
        print("Its a Retinal Image")
    else:
        print("Not Retinal Image")


# In[ ]:




