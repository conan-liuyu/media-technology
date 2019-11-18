import cv2
import numpy as np

def ContrastAlgorithm(rgb_img, contrast=0.5, threshold=0.5):
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    img = img * 1.0
    img_out = img

    if contrast == 1:
        mask_1 = img >= threshold*255.0
        rgb1 = 255.0
        rgb2 = 0
        img_out = rgb1 * mask_1 + rgb2 * (1 - mask_1)

    elif contrast >= 0 :
        alpha = 1 - contrast
        alpha = 1/alpha - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - threshold * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - threshold * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - threshold * 255.0) * alpha

    else:
        alpha = contrast
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - threshold * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - threshold * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - threshold * 255.0) * alpha

    img_out = img_out/255.0
    return img_out

def Process(arg):
    #lsImg = np.zeros(image.shape, np.float32)
    hlsCopy = np.copy(hlsImg)

    l = cv2.getTrackbarPos('lightness', 'image processing')
    s = cv2.getTrackbarPos('saturation', 'image processing')
    c = cv2.getTrackbarPos('contrast', 'image processing')

    #调整亮度
    hlsCopy[:, :, 1] = (1.0 + l / float(l_max)) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1

    #对饱和度进行线性变换
    hlsCopy[:, :, 2] = (1.0 + s / float(s_max)) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1

    lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    lsImg = lsImg*255.0
    lsImg = ContrastAlgorithm(lsImg, c-1)
#    lsImg = cv2.cvtColor(lsImg, cv2.COLOR_RGB2BGR)

    cv2.imshow("image processing", lsImg)

image = cv2.imread('timg.jpg')
(x, y, z) = image.shape

# 图像归一化，且转换为浮点型, 颜色空间转换 BGR转为HLS
fImg = image.astype(np.float32)
fImg = fImg / 255.0
#HLS空间，三个通道分别是: Hue色相、lightness亮度、saturation饱和度
#通道0是色相、通道1是亮度、通道2是饱和度
hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)


lightness = 0
saturation = 0
contrast = 1
l_max = 100
s_max = 100
c_max = 2
cv2.namedWindow("image processing", 0)
cv2.resizeWindow("image processing", (int(x*0.7), int(y*0.85)))
cv2.createTrackbar("lightness", "image processing", lightness, l_max, Process)
cv2.createTrackbar("saturation", "image processing", saturation, s_max, Process)
cv2.createTrackbar("contrast", "image processing", contrast, c_max, Process)


Process(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
