import cv2
import os
import numpy as np
import time
if __name__ == '__main__':

    start = time.time()
    try_use_gpu = True
    imgs = []
    result_name = 'result.jpg'
    '''
    pic_dir = 'pics/'
    # 遍历文件夹读取图片
    for root, dirs, files in os.walk(pic_dir):
        for file in files:
            # print(os.path.join(root, file))
            imgs.append(cv2.imread(os.path.join(root, file)))
    '''
    imgs.append(cv2.imread('1.png'))
    imgs.append(cv2.imread('2.png'))
    # print(imgs)
    # 图片拼接
    stitcher = cv2.createStitcher(try_use_gpu=try_use_gpu)
    retavol, pano = stitcher.stitch(imgs)
    result = cv2.resize(pano, (int(pano.shape[1]/2), int(pano.shape[0]/2)))

    end = time.time()
    print('%.5f s' %(end-start))
    # 显示拼接结果
    cv2.imwrite(result_name, result)
    cv2.imshow("result", result)
    cv2.waitKey(0)


