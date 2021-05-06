# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:45:25 2021

@author: SysAdmin
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import settings


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

    img_path = '../../data/carla-recordings'
    save_path = '../../data/preproc'
    # img_name = 'test_image221411.png'
    for img_name in [i for i in os.listdir(img_path) if i.endswith('.png')]:
        fname = os.path.join(img_path, img_name)

        img = cv2.imread(fname)
        height, width, _ = img.shape
        # src = np.float32([
        #     [304, 203],
        #     [335, 203],
        #     [width, height],
        #     [0, height]])
        # dst = np.float32([
        #     [304, 203],
        #     [335, 203],
        #     [336, height],
        #     [301, height]])
        # src = order_points(src)
        # dst = order_points(dst)

        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)

        edges = cv2.Canny(img, 225, 175)
        lines = cv2.HoughLinesP(edges, 0.5, np.pi/180, 50, minLineLength=120, maxLineGap=50)
        for line in lines:
            x1, y1, x2, y2 = line.squeeze()
            ang = np.arctan2(y2-y1, x2-x1)

            # Remove any lines that aren't mostly vertical
            if (abs(ang) > np.pi/4 and abs(ang) < 3*np.pi/4):
                # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                continue

            # cv2.line(img, (x1, y1), (x2, y2),(255, 0, 0), thickness=2)
            normal = np.array([[-(y2-y1)],
                               [x2-x1]], dtype=np.float32)
            normal /= np.linalg.norm(normal)

            point = np.array([[x1],[y1]], dtype=np.float32)
            outer = normal@normal.T

            Lhs += outer
            Rhs += outer@point

        vanishing_point = np.linalg.inv(Lhs)@Rhs

        top = vanishing_point[1] + 60
        bottom = settings.ORIGINAL_SIZE[1]-35
        width = 530
        def on_line(p1, p2, ycoord):
            return [p1[0]+ (p2[0]-p1[0])/float(p2[1]-p1[1])*(ycoord-p1[1]), ycoord]

        p1 = [vanishing_point[0] - width/2, top]
        p2 = [vanishing_point[0] + width/2, top]
        p3 = on_line(p2, vanishing_point, bottom)
        p4 = on_line(p1, vanishing_point, bottom)
        src = np.array([p1,p2,p3,p4], dtype=np.float32)

        dst = np.array([[0, 0], [settings.UNWARPED_SIZE[0], 0],
                               [settings.UNWARPED_SIZE[0], settings.UNWARPED_SIZE[1]],
                               [0, settings.UNWARPED_SIZE[1]]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (width, height))

        spath = os.path.join(save_path, 'preproc' + img_name)
        cv2.imwrite(spath, warped)

    plt.close('all')

    plt.figure()
    plt.imshow(img)
    plt.title('Before Warp')

    plt.figure()
    plt.imshow(edges)
    plt.title('Edges')

    plt.figure()
    plt.imshow(warped)
    plt.title('After Warp')

"""
    img_path = '../../data/carla-recordings'
    save_path = '../../data/preproc'
    # img_name = 'test_image221411.png'
    for img_name in [i for i in os.listdir(img_path) if i.endswith('.png')]:

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

        spath = os.path.join(save_path, 'preproc' + img_name)
        cv2.imwrite(spath, warped)

        # warped = four_point_transform(img, src)

        # plt.close('all')

        # plt.figure()
        # plt.imshow(img)
        # plt.title('Before Warp')

        # plt.figure()
        # plt.imshow(warped)
        # plt.title('After Warp')
"""