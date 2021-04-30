# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:45:25 2021

@author: SysAdmin
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImgPreprocessor:
    def __init__(self):
        self.src = np.float32([
            [304, 203],
            [335, 203],
            [width, height],
            [0, height]])
        self.dst = np.float32([
            [304, 203],
            [335, 203],
            [336, height],
            [301, height]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        self.img_list = []

    def transform(self, img):
        warped = cv2.warpPerspective(img, M, (width, height))
        return warped

    def get_img(self):
        pass

    def load_img(self, fname):
        self.img_list.append(fname)

    def load_img_dir(self, path):
        self._img_dir = path


# def order_points(pts):
#     # initialzie a list of coordinates that will be ordered
#     # such that the first entry in the list is the top-left,
#     # the second entry is the top-right, the third is the
#     # bottom-right, and the fourth is the bottom-left
#     rect = np.zeros((4, 2), dtype="float32")
#     # the top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#     # now, compute the difference between the points, the
#     # top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     # return the ordered coordinates
#     return rect


# def four_point_transform(image, pts):
#     # obtain a consistent order of the points and unpack them
#     # individually
#     # 	rect = order_points(pts)
#     rect = pts
#     (tl, tr, br, bl) = rect
#     # compute the width of the new image, which will be the
#     # maximum distance between bottom-right and bottom-left
#     # x-coordiates or the top-right and top-left x-coordinates
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))
#     # compute the height of the new image, which will be the
#     # maximum distance between the top-right and bottom-right
#     # y-coordinates or the top-left and bottom-left y-coordinates
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
#     # now that we have the dimensions of the new image, construct
#     # the set of destination points to obtain a "birds eye view",
#     # (i.e. top-down view) of the image, again specifying points
#     # in the top-left, top-right, bottom-right, and bottom-left
#     # order
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")
#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#     # return the warped image
#     return warped


if __name__ == '__main__':
    img_path = '../../data/carla-recordings'
    img_name = 'test_image221411.png'

    fname = os.path.join(img_path, img_name)

    img = cv2.imread(fname)
    height, width, _ = img.shape
    src = np.float32([
        [304, 203],
        [335, 203],
        [width, height],
        [0, height]])
    dst = np.float32([
        [304, 203],
        [335, 203],
        [336, height],
        [301, height]])
    # src = order_points(src)
    # dst = order_points(dst)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (width, height))

    # warped = four_point_transform(img, src)

    plt.close('all')

    plt.figure()
    plt.imshow(img)
    plt.title('Before Warp')

    plt.figure()
    plt.imshow(warped)
    plt.title('After Warp')
