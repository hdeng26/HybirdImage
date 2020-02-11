import numpy as np
import math
import cv2
import sys

from scipy import ndimage


#generate Gaussian Kernel for Gaussian Blur
def gaussianKernel(sigma):
    
    size = math.ceil(3*sigma)
    
    h, k = np.mgrid[-size:size+1, -size:size+1]

    g = (np.exp(-(h*h + k*k)/(2*sigma*sigma)))/(2*np.pi*sigma*sigma)
    return g

#2d convolution
def convolution2d(img, kernel):
    grad = np.array(img)
    size = kernel.shape[0]//2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            gradP = 0
            for h in range(-size,size+1):
                for k in range(-size,size+1):

                    try:

                        gradP += kernel[h, k] * img[i-h, j-k]
                        
                    except:
                        gradP += kernel[h, k] * img[i, j]

            grad[i,j] = gradP

    return grad

#lowpass by three channels
#smooth each channel by gaussian kernel and merge them together
def lowpass(img, sigma):
    A = gaussianKernel(sigma)
    (B, G, R) = cv2.split(img)
    blurB = convolution2d(B, A)
    blurG = convolution2d(G, A)
    blurR = convolution2d(R, A)
    Blur = cv2.merge([blurB, blurG, blurR])
    
    return Blur

#Gaussian Blur(lowpass) used for highpass
def gaussianBlur(img, sigma):
    A = gaussianKernel(sigma)
    return convolution2d(img, A)

#never used but write as requirement said
def correlation2d(img1, img2):
    grad = np.array(img1)
    size = img2.shape[0]//2
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            
            gradP = 0
            for h in range(-size,size+1):
                for k in range(-size,size+1):
   
                    try:
                 
                        gradP += img2[h, k] * img1[i+h, j+k]
                        
                    except:
                        gradP += img2[h, k] * img1[i, j]

            grad[i,j] = gradP

    return hyd

# highpass function
def highpass(img, sigma):
    #use original image - lowpass = highpass
    hp = np.array(img)
    lowpass = gaussianBlur(img,sigma)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hp[i, j] = int(img[i,j])- int(lowpass[i,j]) + 128
    return hp

#sigma1 is for img1(lowpass sigma) and sigma2 is for img2(highpass sigma)
def hybrid(img1, img2, sigma1, sigma2):

    #set three channels for iamge 1 for split(same format as img2)
    hybrid1 = np.array(img2)
    hybrid2 = np.array(img2)
    hybrid3 = np.array(img2)
    lp = lowpass(img1,sigma1)
    
    hp = highpass(img2, sigma2)

    #split three channels and add highpass to all of them
    (B, G, R) = cv2.split(lp)

    
    #channel blue
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            hybrid1[i, j] = int(hp[i,j]) + int(B[i,j])-128
    #channel green
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            hybrid2[i, j] = int(hp[i,j]) + int(G[i,j])-128
    #channel red
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            hybrid3[i, j] = int(hp[i,j]) + int(R[i,j])-128

    #merge all chennals and get final hybrid image
    hybrid = cv2.merge([hybrid1, hybrid2, hybrid3])
    
    return hybrid


# main functions
img1 = cv2.imread('left.jpg',1)
img2 = cv2.imread('right.jpg',0)

# hybrid(lowpass_image, highpass_image, lowpass_sigma, highpass_sigma)
hyd = hybrid(img1, img2, 2, 1)

cv2.imshow("hybrid", hyd)
cv2.imwrite("hybrid.jpg", hyd)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

    
