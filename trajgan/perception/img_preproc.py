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
    def __init__(self, img_format='.png', warp_size=(500, 500)):
        """Image preprocessing class for

        Reference: https://github.com/Ayanzadeh93/Udacity-Advance-Lane-detection-of-the-road

        Parameters
        ----------
        img_format : str, optional
            Image format of the images being read. The default is '.png'.

        warp_size : tuple(int, int), optional
            Final size of the warped image in pixels. The default is (500, 500).

        Returns
        -------
        None.

        """
        self._format = img_format
        self._warp_size = warp_size

        self.img_list = []
        self._cur_img = None

    def transform(self, img, return_xform_mat=False, return_vp=False):
        height, width = self._warp_size
        Lhs = np.zeros((2,2), dtype= np.float32)
        Rhs = np.zeros((2,1), dtype= np.float32)

        # Detect lane lines and find the vanishing point
        edges = cv2.Canny(img, 225, 175)
        lines = cv2.HoughLinesP(edges, 0.5, np.pi/180, 20, minLineLength=120, maxLineGap=50)
        for line in lines:
            x1, y1, x2, y2 = line.squeeze()
            ang = np.arctan2(y2-y1, x2-x1)

            # Remove any lines that aren't mostly vertical
            if (abs(ang) > np.pi/4 and abs(ang) < 3*np.pi/4): continue

            normal = np.array([[-(y2-y1)],
                               [x2-x1]], dtype=np.float32)
            normal /= np.linalg.norm(normal)

            point = np.array([[x1],[y1]], dtype=np.float32)
            outer = normal@normal.T

            Lhs += outer
            Rhs += outer@point

        vanishing_point = (np.linalg.inv(Lhs)@Rhs).squeeze()

        # Given the vanishing point, determine what the ideal camera transform would be
        top = vanishing_point[1] + 60
        bottom = height - 20
        def on_line(p1, p2, ycoord):
            return [p1[0]+ (p2[0] - p1[0]) / float(p2[1] - p1[1])*(ycoord-p1[1]), ycoord]

        p1 = [vanishing_point[0] - width/2, top]
        p2 = [vanishing_point[0] + width/2, top]
        p3 = on_line(p2, vanishing_point, bottom)
        p4 = on_line(p1, vanishing_point, bottom)
        src = np.array([p1,p2,p3,p4], dtype=np.float32)

        dst = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]], dtype=np.float32)

        # Compute the forward and reverse perspective transform and apply it
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (width, height))

        if return_xform_mat:
            return warped, M, Minv
        elif return_vp:
            return warped, vanishing_point, 
        else:
            return warped

    def load_img_path(self, fname):
        self.img_list.append(fname)

    def load_img_dir(self, path):
        self._img_dir = path
        new_imgs = [i for i in os.listdir(path) if i.endswith(self._format)]
        self.img_list += new_imgs

    def preproc_batch(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for img_name in self.img_list:
            try:
                img = cv2.imread(os.path.join(self._img_dir, img_name))
                xform = self.transform(img)
                save_name = os.path.join(save_path, 'preproc_' + img_name)
                cv2.imwrite(save_name, xform)
            except Exception as e:
                print(f'[!] Issue with {img_name}')
                print(e)


if __name__ == '__main__':
    img_path = '../../data/carla-recordings/out9'
    img_name = '4112.png'
    save_path = '../../data/preproc/trin_data'

    preproc = ImgPreprocessor()
    preproc.load_img_dir(img_path)
    # preproc.preproc_batch(save_path) preproc_image_test017257

