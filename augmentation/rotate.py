import cv2
import numpy as np
import os
from shapely.geometry import Polygon, box
from .base import AugmentBase
import copy
import random


class Rotate(AugmentBase):
    def __init__(self):
        super().__init__()
        
    def __call__(self, **kwargs):
        """

        """
        
        # define hyperparameter
        angles = kwargs['angle']
        padding = kwargs.get('padding', 0)
        
        if len(angles) == 1:
            angle = angles[0]
            
        elif len(angles) == 2:
            if angles[0] >= angles[1]:
                f_angle = angles[1]
                s_angle = angles[0]
            else:
                f_angle = angles[0]
                s_angle = angles[1]
            
            angle = random.randint(f_angle, s_angle)
        
        else:
            raise ValueError(f"angle value is only one or two")
        
        # Get image
        origin_img = self.image.copy()
        origin_img_h, origin_img_w = origin_img.shape[:2]
        
        # Use padding to overcome the OBB coordinates are outside the image in rotation.
        padded_img, padded_img_h, padded_img_w = self.padding_img(origin_img, padding)
        padded_cx, padded_cy = padded_img_w // 2, padded_img_h // 2  # 회전 중심
        
    
        # Use padding to obbs
        
        obbs = copy.deepcopy(self.oriented_bounding_boxes)
        padding_label = self.padding_label(obbs)
        
        # Rotation Process
        rotation_info_dict = {}
        # for idx in range(self.iter_num):
        rotate_matrix = cv2.getRotationMatrix2D((padded_cx, padded_cy), angle, 1.0) # Generate Rotation Matrix 
        rotated_image = cv2.warpAffine(padded_img, rotate_matrix, (padded_img_w, padded_img_h))   # Rotate image
        save_img_name = self.image_name # + f"_{idx}"
        # 회전된 OBB 계산
        rotated_xywha, rotated_xyxyxy = self.rotate_obb(padding_label, angle, padded_img_w, padded_img_h, rotate_matrix)
        
        if save_img_name not in rotation_info_dict:
            rotation_info_dict[save_img_name] = {
                "image" : rotated_image,
                "xyxyxyxy" : rotated_xyxyxy,
                "xywha" : rotated_xywha 
            }
                
        # save data & visualize
        # for img_name, info in rotation_info_dict.items():
        #     mode = 'rotate'
        #     # save image
        #     self.save_img(img_name, info, mode)
        #     # save xywhr format
        #     self.save_xywhr(img_name, info, mode)
        #     # save xyxyxyxy format
        #     self.save_xyxyxyxy(img_name, info, padded_img_w, padded_img_h, mode)
        #     # visualize
        #     self.visualize(img_name, info, mode)
                
            
        return rotation_info_dict
            
    def rotate_obb(self, obbs, angle, img_w, img_h, rotate_matrix):
        rotated_xywha = []
        rotated_xyxyxyxy = []
        
        for idx, obb in enumerate(obbs):
            if len(obb) == 6:
                cls, x, y, w, h, a = obb
                # rotate obb using rotate matrix
                corners = self.xywha_xyxyxyxy(x, y, w, h, a)
            elif len(obb)==2:
                cls = obb[0]
                corners = obb[1].reshape(4,2)
            
            else:
                print("OBB 형식이 맞지 않습니다.")
                return None    
            
            rotated_corners = self._rotate_corners(corners, rotate_matrix)
            intersection_area_ratio, intersection_coords, \
                img_polygon = self.calculate_intersection_area(rotated_corners, 0, 0, img_w, img_h)  # calculate intersection area between image and obb
            
            # Remove or Modify obbs process
            if self._should_skip_box(intersection_area_ratio):
                # print(f"❌ 제거된 좌표: idx : {idx}, x : {x}, y : {y}, w : {w}, h : {h}, a : {a}, area_ratio : {area_ratio:.2f}")
                continue
            
            epsilon = 1e-6
            if intersection_area_ratio < 1-epsilon:  # 완전 안 맞을 경우
                # print(f"줄여야되는 좌표: idx: {idx}, x: {x}, y: {y}, w: {w}, h: {h}, a: {a}, area_ratio: {intersection_area_ratio:.2f}")
                rotated_corners= self.fix_obb_with_min_area_rect(intersection_coords)
                if rotated_corners is None:
                    continue
            
            x_coords, y_coords = rotated_corners[:, 0], rotated_corners[:, 1]   # extract each (x,y) coordinates
            
            # save coordinates xyxyxyxy 
            rotated_corners = rotated_corners.flatten()         # shape (8,)
            rotated_xyxyxyxy.append((cls, rotated_corners)) 

            # change xyxyxyxy to xywha
            # x_new, y_new = x_coords.mean(), y_coords.mean()
            # a_new = self._normalize_angle(a + angle)
            # rotated_xywha.append((cls, x_new, y_new, w, h, a_new))

        return rotated_xywha, rotated_xyxyxyxy


    def _rotate_corners(self, corners, rotate_matrix):
        """회전 행렬 적용"""
        ones = np.ones((4, 1))
        homogeneous = np.hstack([corners, ones]).T
        return np.dot(rotate_matrix, homogeneous).T[:, :2]


    def _normalize_angle(self, angle):
        """-90~90 범위로 정규화"""
        if angle > 90:
            return angle - 180
        elif angle < -90:
            return angle + 180
        return angle


    

    
    
        
        
        
        
        



