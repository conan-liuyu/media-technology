import cv2
import numpy as np
#import matplotlib as plt

def filtering(arg):
    Img = np.copy(img)

    m1 = cv2.getTrackbarPos('mean', 'image filtering')
    G = cv2.getTrackbarPos('Gaussian', 'image filtering')
    m2 = cv2.getTrackbarPos('median', 'image filtering')
    b = cv2.getTrackbarPos('bilater', 'image filtering')
    sobel = cv2.getTrackbarPos('sobel', 'image filtering')

    img_gray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

    if(m1):
        Img = cv2.blur(Img, (5, 5))     # 均值滤波
    if(G):
        Img = cv2.GaussianBlur(Img, (5,5), 0)      # 高斯滤波

    if(m2):
        Img = cv2.medianBlur(Img, 5)     #中值滤波
    if(b):
        Img = cv2.bilateralFilter(Img, 9, 75, 75)       #双边滤波
    if(sobel):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        absX = cv2.convertScaleAbs(sobelx)   # 转回uint8
        absY = cv2.convertScaleAbs(sobely)
        Img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


    cv2.imshow('image filtering', Img)

img = cv2.imread('test..jpg')
(x, y, z) = img.shape
cv2.namedWindow('image filtering', 0)
cv2.resizeWindow("image filtering", (int(x*0.7), int(y*1.3)))
s = [0, 0, 0, 0, 0, 0]
s_max = [1, 1, 1, 1, 1, 1]
cv2.createTrackbar("mean", "image filtering", s[0], s_max[0], filtering)
cv2.createTrackbar("Gaussian", "image filtering", s[1], s_max[1], filtering)
cv2.createTrackbar("median", "image filtering", s[2], s_max[2], filtering)
cv2.createTrackbar("bilater", "image filtering", s[3], s_max[3], filtering)
cv2.createTrackbar("sobel", "image filtering", s[4], s_max[4], filtering)


filtering(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
