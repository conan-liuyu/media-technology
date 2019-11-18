import numpy as np
import cv2
import time
import imutils

def load_image(path, gray=False):
    if gray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.imread(path)
def transform(origin):
    h, w, _ = origin.shape
    ini = np.zeros(origin.shape)
    for i in range(h):
        for j in range(w):
            ini[i, w - 1 - j] = origin[i, j]
    return ini.astype(np.uint8)

def extract_match(img1, img2):
    # 实例化
    sift = cv2.xfeatures2d.SIFT_create()

    # 计算关键点和描述子, kp为关键点keypoints, des为描述子descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 绘出关键点, 其中参数分别是源图像、关键点、输出图像、显示颜色
    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 255))
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 255))

    cv2.imshow('1', img3)
    cv2.imshow('2', img4)
    # 参数设计和实例化
    index_params = dict(algorithm=1, trees=6)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 利用knn计算两个描述子的匹配
    matche = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matche))]

    # 绘出匹配效果
    result = []
    for m, n in matche:
        if m.distance < 0.6 * n.distance:
            result.append([m])

    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matche, None, flags=2)
    cv2.imshow("MatchResult", img5)
    cv2.waitKey(0)

def fun(imageA, imageB):
    class Stitcher:
        def __init__(self):
            # determine if we are using OpenCV v3.X
            self.isv3 = imutils.is_cv3()

        def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
            # unpack the images, then detect keypoints and extract
            # local invariant descriptors from them

            (imageB, imageA) = images
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                return None

            # otherwise, apply a perspective warp to stitch the images
            # together
            (matches, H, status) = M
            result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

            # check to see if the keypoint matches should be visualized
            if showMatches:
                vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                    status)
                # return a tuple of the stitched image and the
                # visualization
                return (result, vis)

            # return the stitched image
            return result

        #接收照片，检测关键点和提取局部不变特征
        #用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
        #detectAndCompute方法用来处理提取关键点和特征
        #返回一系列的关键点
        def detectAndDescribe(self, image):
            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # check to see if we are using OpenCV 3.X
            if self.isv3:
                # detect and extract features from the image
                descriptor = cv2.xfeatures2d.SIFT_create()
                (kps, features) = descriptor.detectAndCompute(image, None)

            # otherwise, we are using OpenCV 2.4.X
            else:
                # detect keypoints in the image
                detector = cv2.FeatureDetector_create("SIFT")
                kps = detector.detect(gray)

                # extract features from the image
                extractor = cv2.DescriptorExtractor_create("SIFT")
                (kps, features) = extractor.compute(gray, kps)

            # convert the keypoints from KeyPoint objects to NumPy
            # arrays
            kps = np.float32([kp.pt for kp in kps])

            # return a tuple of keypoints and features
            return (kps, features)
        #matchKeypoints方法需要四个参数，第一张图片的关键点和特征向量，第二张图片的关键点特征向量。
        #David Lowe’s ratio测试变量和RANSAC重投影门限也应该被提供。
        def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
            ratio, reprojThresh):
            # compute the raw matches and initialize the list of actual
            # matches
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
            matches = []

            # loop over the raw matches
            for m in rawMatches:
                # ensure the distance is within a certain ratio of each
                # other (i.e. Lowe's ratio test)
                if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                    matches.append((m[0].trainIdx, m[0].queryIdx))

            # computing a homography requires at least 4 matches
            if len(matches) > 4:
                # construct the two sets of points
                ptsA = np.float32([kpsA[i] for (_, i) in matches])
                ptsB = np.float32([kpsB[i] for (i, _) in matches])

                # compute the homography between the two sets of points
                (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                    reprojThresh)

                # return the matches along with the homograpy matrix
                # and status of each matched point
                return (matches, H, status)

            # otherwise, no homograpy could be computed
            return None
        #连线画出两幅图的匹配
        def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
            # initialize the output visualization image
            (hA, wA) = imageA.shape[:2]
            (hB, wB) = imageB.shape[:2]
            vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
            vis[0:hA, 0:wA] = imageA
            vis[0:hB, wA:] = imageB

            # loop over the matches
            for ((trainIdx, queryIdx), s) in zip(matches, status):
                # only process the match if the keypoint was successfully
                # matched
                if s == 1:
                    # draw the match
                    ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                    ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                    cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

            # return the visualization
            return vis

    # load the two images and resize them to have a width of 400 pixels
    # (for faster processing)
    #imageA = cv2.imread('1.png')
    #imageB = cv2.imread('2.png')
    imageA = cv2.resize(imageA, (imageB.shape[0], imageB.shape[1]))
    # stitch the images together to create a panorama
    # showMatches=True 展示两幅图像特征的匹配,返回vis
    start = time.time()
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    # show the images
    end = time.time()
    print('%.5f s' %(end-start))
    cv2.imshow('vis', vis)
    cv2.imshow('result', result)
    cv2.waitKey(0)

if __name__ == '__main__':
    img1 = cv2.imread('1.png')
    img2 = cv2.imread('2.png')
    #extract_match(img1, img2)
    fun(img1, img2)


