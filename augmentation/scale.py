import random
import numpy as np
import cv2
import os
from shapely.geometry import Polygon, box
import copy
import random
from .base import AugmentBase
import random

class Scaling(AugmentBase):
    def __init__(self):
        super().__init__()
    
    def __call__(self, **kwargs):
        
        # define hyperparameter
        scales = kwargs['scale']
        padding = kwargs.get('padding', 0)
    
        # choose scales value
        if len(scales) > 1:
            scale = random.choice(scales)
        else:
            scale = scales[0]
        
        # scales : list, scale : tuple (scale_x, scale_y)
        scale_x, scale_y = scale[0], scale[1]
        
        assert 0 < scale_x and 0 < scale_y, "scale value is always bigger than 0."
        
        # Load image
        origin_img = self.image.copy()
        origin_img_h, origin_img_w = origin_img.shape[:2]
        obbs = copy.deepcopy(self.oriented_bounding_boxes)

        # Scaling Process
        scaled_info_dict = {}
        scale_matrix = self.get_scaling_affine_matrix(scale_x, scale_y, origin_img_h, origin_img_w)
        scale_img = cv2.warpAffine(origin_img, scale_matrix, (origin_img_w, origin_img_h), flags=cv2.INTER_LINEAR)
        save_img_name = self.image_name
        
        scaled_xywha, scaled_xyxyxyxy = self.scale_obb(obbs, scale_img, scale_matrix)
        
        if save_img_name not in scaled_info_dict:
            scaled_info_dict[save_img_name] = {
                "image" : scale_img,
                "xyxyxyxy" : scaled_xyxyxyxy,
                "xywha" : scaled_xywha
            }
            
        return scaled_info_dict
        
        
        
    def get_scaling_affine_matrix(self, scale_x, scale_y, img_h, img_w):
        
        cx, cy = img_w/2, img_h/2
        
        # Scaling based on center
        # M = cv2.getRotationMatrix2D(center=(cx,cy), angle=0, scale=1.0) 
        # M[0, 0] *= scale_x
        # M[0, 1] *= scale_x
        # M[1, 0] *= scale_y
        # M[1, 1] *= scale_y
        
        # Step 1: 중심 기준 이동 -> 스케일 -> 다시 원래 위치로
        # Affine 행렬 직접 구성
        translate1 = np.array([[1, 0, -cx],
                            [0, 1, -cy],
                            [0, 0, 1]])

        scale = np.array([[scale_x, 0, 0],
                        [0, scale_y, 0],
                        [0, 0, 1]])

        translate2 = np.array([[1, 0, cx],
                            [0, 1, cy],
                            [0, 0, 1]])

        # 전체 affine 행렬: T2 @ S @ T1
        affine_matrix = translate2 @ scale @ translate1

        return affine_matrix[:2, :]  # cv2.warpAffine()에 맞춰 (2,3) 형태로 리턴
        
        
       
    
    
    def scale_obb(self, obbs, scale_img, scale_matrix):
        scaled_xywha = []
        scaled_xyxyxyxy = []
        
        img_h, img_w = scale_img.shape[:2]
        
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
            
            scaled_corners = self._scale_corners(corners, scale_matrix)
            
            intersection_area_ratio, intersection_coords, \
                img_polygon = self.calculate_intersection_area(scaled_corners, 0, 0, img_w, img_h)  # calculate intersection area between image and obb
            
            if self._should_skip_box(intersection_area_ratio):
                # print(f"❌ 제거된 좌표: idx : {idx}, x : {x}, y : {y}, w : {w}, h : {h}, a : {a}, area_ratio : {area_ratio:.2f}")
                continue
            
            epsilon = 1e-6
            if intersection_area_ratio < 1-epsilon:  # 완전 안 맞을 경우
                # print(f"줄여야되는 좌표: idx: {idx}, x: {x}, y: {y}, w: {w}, h: {h}, a: {a}, area_ratio: {intersection_area_ratio:.2f}")
                scaled_corners= self.fix_obb_with_min_area_rect(intersection_coords)
                if scaled_corners is None:
                    continue
            
            # save coordinates xyxyxyxy 
            scaled_corners = scaled_corners.flatten()         # shape (8,)
            scaled_xyxyxyxy.append((cls, scaled_corners))
            
        return scaled_xywha, scaled_xyxyxyxy 
            
            
    def _scale_corners(self, corners, scale_matrix):
        """스케일 행렬 적용"""
        ones = np.ones((4,1))
        homogeneous = np.hstack([corners, ones]).T  # (3, 4)
        return np.dot(scale_matrix, homogeneous).T[:, :2]  # (4, 2)
        
        
        
    
    
        

        