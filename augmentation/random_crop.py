import random
import numpy as np
import cv2
import os
from shapely.geometry import Polygon, box
import copy
import random
from .base import AugmentBase




class RandomCrop(AugmentBase):
    def __init__(self):
        super().__init__()
    
    def __call__(self, scale, padding=0):
        
        assert 0 < scale <= 1, "scale은 (0,1] 사이여야 합니다."
        crop_options = ['top_left','top_right', 'bottom_left', 'bottom_right', 'random']
        
        # Get image & labels
        origin_img = self.image.copy()
        origin_img_h, origin_img_w = origin_img.shape[:2]
        obbs = copy.deepcopy(self.oriented_bounding_boxes)
       
        # image 크롭 및 저장
        cropped_img_dict = {}
        
        crop_mode = 'random'# random.choice(crop_options)
        cropped_img, cropped_coordinates = self.crop_image(origin_img, scale, crop_mode)
        save_img_name = self.image_name # + f"_{idx}"
        
        cropped_xyxyxyxy = self.process_obbs_for_crop(cropped_coordinates, obbs)
    
        if save_img_name not in cropped_img_dict:
            cropped_img_dict[save_img_name] = {
                "image" : cropped_img,
                "xyxyxyxy" : cropped_xyxyxyxy
            }
                
        # save data & visualize
        # for img_name, info in cropped_img_dict.items():
        #     img_h, img_w = info['image'].shape[:2] 
        #     mode = 'crop'
        #     # save image
        #     self.save_img(img_name, info, mode)
        #     # save xyxyxyxy format
        #     self.save_xyxyxyxy(img_name, info, img_w, img_h, mode)
        #     # visualize
        #     self.visualize(img_name, info, mode)
            
        return cropped_img_dict
                    
        
        
    
    def crop_image(self, image, scale, crop_mode):
        x1, y1, x2, y2 = self.get_crop_region(image.shape, scale, crop_mode)
        cropped = image[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)


    def get_crop_region(self, img_shape, scale, crop_mode):
        h, w = img_shape[:2]
        crop_w = min(int(w * scale), w)
        crop_h = min(int(h * scale), h)
        
        if crop_mode == 'top_left':
            start_x = 0
            end_x = max(0, w // 2 - crop_w)
            start_y = 0
            end_y = max(0, h // 2 - crop_h)
        elif crop_mode == 'top_right':
            start_x = w // 2
            end_x = max(0, w - crop_w)
            start_y = 0
            end_y = max(0, h // 2 - crop_h)
        elif crop_mode == 'bottom_left':
            start_x = 0
            end_x = max(0, w // 2 - crop_w)
            start_y = h // 2
            end_y = max(0, h - crop_h)
        elif crop_mode == 'bottom_right':
            start_x = w // 2
            end_x = max(0, w - crop_w)
            start_y = h // 2
            end_y = max(0, h - crop_h)
        elif crop_mode == 'random':
            start_x = 0
            end_x = max(0, w - crop_w)
            start_y = 0
            end_y = max(0, h - crop_h)
        else:
            raise ValueError(f"Invalid mode: {crop_mode}")

        # 여기서 보정
        if start_x > end_x:
            region_x = start_x
        else:
            region_x = random.randint(start_x, end_x)

        if start_y > end_y:
            region_y = start_y
        else:
            region_y = random.randint(start_y, end_y)

        x1 = region_x
        y1 = region_y
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        return x1, y1, x2, y2

    
    
    def process_obbs_for_crop(self, cropped_coordinates, obbs):
        
        x1, y1, x2, y2 = cropped_coordinates
        cropped_xywha = []          # """구현 필요"""
        cropped_xyxyxyxy = []

        for idx, obb in enumerate(obbs):
            
            if len(obb) == 6:
                cls, x, y, w, h, a = obb
                # rotate obb using rotate matrix
                corners = self.xywha_xyxyxyxy(x, y, w, h, a)
            elif len(obb)==2:
                cls = obb[0]
                corners = obb[1].reshape(4,2)
                
            intersection_area_ratio, intersection_coords, \
                img_polygon = self.calculate_intersection_area(corners, x1, y1, x2, y2)
                
            # Remove or Modify obbs process
            if self._should_skip_box(intersection_area_ratio):
                # print(f"❌ 제거된 좌표: idx : {idx}, x : {x}, y : {y}, w : {w}, h : {h}, a : {a}, area_ratio : {area_ratio:.2f}")
                continue    
            
            epsilon = 1e-6
            if intersection_area_ratio < 1-epsilon:  # 완전 안 맞을 경우
                # print(f"줄여야되는 좌표: idx: {idx}, x: {x}, y: {y}, w: {w}, h: {h}, a: {a}, area_ratio: {intersection_area_ratio:.2f}")
                corners= self.fix_obb_with_min_area_rect(intersection_coords)
                if corners is None:
                    continue

            corners -= np.array([x1, y1])   # Move Obb from origin to crop images (0,0) 
            x_coords, y_coords = corners[:, 0], corners[:, 1]   # extract each (x,y) coordinates
            
            # save coordinates xyxyxyxy 
            corners = corners.flatten()         # shape (8,)
            cropped_xyxyxyxy.append((cls, corners)) 
            
        return cropped_xyxyxyxy
            
            
            
    
    


