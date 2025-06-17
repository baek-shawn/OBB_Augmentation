import random
import numpy as np
import cv2
import os
from shapely.geometry import Polygon, box
import copy
import random
from .base import AugmentBase
import random



class Tiling(AugmentBase):
    def __init__(self):
        super().__init__()
    
    def __call__(self, **kwargs):
        """
        parameters
        scale : (slice_width_ratio, slice_height_ratio)
        overlap_ratio : (overlap_width_ratio, overlap_height_ratio)
        """
        # define hyperparameter
        scales = kwargs['scale']
        overlap_ratio = kwargs['overlap_ratio'] if kwargs['overlap_ratio'] else [0,0]# overlap ratio must has only one tuple
        fix_mode = kwargs['fix_mode']      # ['fix', 'dynamic', 'None']
        model_size = kwargs['model_size'] if kwargs['model_size'] else 640
        
        # choose scale value
        if len(scales) > 1:
            scale = random.choice(scale)
        else:
            scale = scales[0]
        
        
        # scales : list, scale : tuple (scale_x, scale_y)
        slice_x, slice_y = scale[0], scale[1]
        overlap_x, overlap_y = overlap_ratio[0], overlap_ratio[1]
        assert 0 < slice_x and 0 < slice_y, "scale value is always bigger than 0."
        assert 0 < overlap_x and 0 < overlap_y, "overlap value is always bigger than 0."
        
        # Get image & labels
        tiled_img_dict = {}
        for idx, (img, oriented_bboxes, img_name) in enumerate(zip(self.image, self.oriented_bounding_boxes, self.image_name)):
            origin_img = img.copy()
            origin_img_h, origin_img_w = origin_img.shape[:2]
            obbs = copy.deepcopy(oriented_bboxes)
        
            # image slicing
            tiled_data = self.generate_tiles(origin_img, slice_x, slice_y, overlap_x, overlap_y, fix_mode, model_size)
            
            
            for idx, (tile_coords, tile_img) in enumerate(tiled_data):
                cropped_obbs = self.process_obbs_for_crop(tile_coords, obbs)
                if not cropped_obbs:
                    continue
                save_img_name = f"{img_name}_tile_{idx}"
                tiled_img_dict[save_img_name] = {
                "image": tile_img,
                "xyxyxyxy": cropped_obbs
            }
    
        return tiled_img_dict
                    
        
    def generate_tiles(self, image, slice_x, slice_y, overlap_x, overlap_y, fix_mode, model_size):
        """
        slice tiles
        """
        img_h, img_w = image.shape[:2]
        
        slice_w, slice_h = int(img_w*slice_x), int(img_h*slice_y)
        
        # When applying parameters that fix the image size (usually used to increase or decrease the size of the slice image)
        if fix_mode == "fix":
            if model_size <= img_w and model_size <= img_h:
                slice_w, slice_h = model_size, model_size
            else:
                model_size = min(img_w, img_h)
                slice_w, slice_h = model_size, model_size
           
        elif fix_mode == "dynamic":
            if slice_w <= model_size or slice_h <= model_size:
                slice_w, slice_h = model_size, model_size
                
        stride_w, stride_h = int(slice_w*(1-overlap_x)), int(slice_h*(1-overlap_y))
        
        tiles = []
        for y in range(0, img_h, stride_h):
            for x in range(0, img_w, stride_w):
                x1 = min(x, img_w - slice_w)
                y1 = min(y, img_h - slice_h)
                x2 = x1 + slice_w
                y2 = y1 + slice_h
                cropped = image[y1:y2, x1:x2]
                tiles.append(((x1, y1, x2, y2), cropped))
    
        return tiles
        
        
    
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
            
            
            
    
    


