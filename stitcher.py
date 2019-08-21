import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


im0 = mpimg.imread("6_camera_align/align0.jpg")
im1 = mpimg.imread("6_camera_align/align1.jpg")
plt.imshow(im1)

border_width = 0
# pad images
im0 = cv2.copyMakeBorder(im0,border_width,border_width,border_width,border_width,cv2.BORDER_CONSTANT,(0,0,0))

im_pts_0 = np.array([[3759,1395],
                    [3145,1028],
                    [3262,314],
                    [3760,697]])
im_pts_1 = np.array([[1233,1716],
                     [240,1384],
                     [146,177],
                     [990,703]])
im_pts_0 = np.float32(im_pts_0+border_width)
im_pts_1 = np.float32(im_pts_1+border_width)

M1 = cv2.getPerspectiveTransform(im_pts_0,im_pts_1)
im_warped = cv2.warpPerspective(im0,M1,(3840,2160))
newim = im_warped + im1

im_warped = cv2.resize(im_warped,(500,500))
newim = cv2.resize(newim,(1920,1080))
cv2.imshow("frame",newim)
cv2.waitKey(0)


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result
#
#dst_pts = float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#src_pts = float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

result = warpTwoImages(im0, im1, M1)
cv2.imshow("frame",result)
