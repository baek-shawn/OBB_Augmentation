import random
import numpy as np
import cv2
import os
import copy
from shapely.geometry import Polygon, box



class AugmentBase:
    """
    Base class for image augmentation.
    """
    def __init__(self):
        """
        Initialize the AugmentBase class with the provided keyword arguments.
        Args:
            **kwargs: Keyword arguments containing the following keys:
        #         - image: The input image to be augmented.
        #         - oriented_bounding_boxes: A list of tuples representing the oriented bounding boxes.
        #         - save_img_path: The path where the augmented image will be saved.
        #         - save_xywhr_path: The path where the augmented bounding box coordinates will be saved.
        #         - save_xyxyxyxy_path: The path where the augmented bounding box coordinates will be saved.
        #         - save_visualize_path: The path where the augmented image with bounding boxes will be saved.
        #         - iter_num : Augmentation repeat's number (Remove ver)
        #         - image_name : for save file
         """
        # self.image = kwargs.get("image")
        # self.oriented_bounding_boxes = kwargs.get("oriented_bounding_boxes")
        # self.save_img_path = kwargs.get("save_img_path")
        # self.save_xywhr_path = kwargs.get("save_xywhr_path")
        # self.save_xyxyxyxy_path = kwargs.get("save_xyxyxyxy_path")
        # self.save_visualize_path = kwargs.get("save_visualize_path")
        # self.iter_num = kwargs.get("iter_num", 0)
        # self.image_name = kwargs.get("image_name")
        
    
    def set_data(self, data):
        self.image = data["image"]
        self.oriented_bounding_boxes = data['labels']
        self.image_name = data["image_name"]
        self.save_dirs = data["save_dirs"]
        

    def __call__(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def xywha_xyxyxyxy(self,x,y,w,h,angle):
        """
        Convert oriented bounding box coordinates from (x, y, w, h, angle) to (x1, y1, x2, y2, x3, y3, x4, y4).
        Args: x, y, w, h, angle: The coordinates of the bounding box in the format (x, y, w, h, angle).
        """
        dx, dy = w / 2, h / 2
        angle_rad = np.radians(angle)

        corners = np.array([
            [-dx, -dy],
            [ dx, -dy],
            [ dx,  dy],
            [-dx,  dy]
        ])

        # 회전 변환 행렬 적용
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])

        rotated_corners = np.dot(corners, rotation_matrix) 
        rotated_corners+= np.array([x, y])
        
        return rotated_corners
    
    
    
    def calculate_intersection_area(self, corners, x1, y1, x2, y2):
        """
        Calculate the intersection area between the oriented bounding box and the image or crop bounding box.
        Args:
        - corners: The coordinates of the oriented bounding box in the format (x1, y1, x2, y2, x3, y3, x4, y4).
        - x1, y1, x2, y2: The coordinates of the image or crop bounding box.
        """
        try:
            obb_polygon = Polygon(corners)
            img_polygon = box(x1, y1, x2, y2)  # img 전체 또는 crop 영역을 polygon으로

            intersection = obb_polygon.intersection(img_polygon)

            if intersection.geom_type == 'Polygon':
                obb_area = obb_polygon.area
                intersection_area = intersection.area if not intersection.is_empty else 0
                intersection_ratio = intersection_area / obb_area if obb_area > 0 else 0
                intersection_coords = np.array(intersection.exterior.coords[:-1])
                
                return intersection_ratio, intersection_coords, img_polygon
            else:
                return 0, None, None
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return 0, None, None
        
    def fix_obb_with_min_area_rect(self, intersection_coords):
        """
        Fix the oriented bounding box using the minimum area rectangle.
        Args:
        - intersection_coords: The coordinates of the intersection polygon.
       
        """
        try:
            min_area_rect = cv2.minAreaRect(intersection_coords.astype(np.float32))
            box = cv2.boxPoints(min_area_rect)  # shape (4,2)
         
            return box
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return 0, None, None
        
    def padding_img(self, img, padding=0):
        """
        Pad the image with white (255).
        Args:
        - img: The input image to be padded.
        - padding: The amount of padding to be applied to each side of the image.
        """
        try:
            if padding == 0:
                img_height, img_width = img.shape[:2]
                return img, img_height, img_width
            
            padded_img = cv2.copyMakeBorder(img, top = padding, 
                                            bottom = padding, left = padding, 
                                            right = padding, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
            img_height, img_width = img.shape[:2]
            return padded_img, img_width, img_height
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return None
        
    def padding_label(self, obbs, padding=0):
        """
        Pad the image with white (255).
        Args:
        - obbs: The input Oriented Bounding boxes. The type is numpy array.
        - padding: The amount of padding to be applied to each side of the image.
        """
        if padding == 0:
            return obbs
        try:
            
            obbs[:, 1] += padding
            obbs[:, 2] += padding
        
            return obbs
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return None
        
    def _should_skip_box(self, area_ratio, threshold=0.4):
        """박스를 건너뛰어야 하는지 판단"""
        return area_ratio < threshold
        
    def save_img(self, save_dir, img_name, info):
        img_name = img_name + '.png'
        img_path = os.path.join(save_dir, img_name)
        image = copy.deepcopy(info["image"])
        cv2.imwrite(img_path, image)
        print(f"✅ Augmentation 이미지 저장: {img_path}")    
    
        
    def save_xywhr(self, save_dir, img_name, info):
        # save xywhr format
        img_name = img_name+".txt"
        xywhr_path = os.path.join(save_dir, img_name)
        with open(xywhr_path, "w") as f:
            f.write("YOLO_OBB\n")
            for obb in info["xywha"]:
                cls, x, y, w, h, a = obb
                f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {a:.6f}\n")
                
        print(f"✅ Augmentation xywhr 라벨 저장: {xywhr_path}")
                
    def save_xyxyxyxy(self, save_dir, img_name, info, img_w, img_h):
        img_name = img_name+".txt"
        xyxyxyxy_path = os.path.join(save_dir, img_name)
        with open(xyxyxyxy_path, "w") as f:
            for cls, xyxyxyxy in info['labels']:
                x1,x2,x3,x4 = xyxyxyxy[0]/img_w, xyxyxyxy[2]/img_w, xyxyxyxy[4]/img_w, xyxyxyxy[6]/img_w
                y1,y2,y3,y4 = xyxyxyxy[1]/img_h, xyxyxyxy[3]/img_h, xyxyxyxy[5]/img_h, xyxyxyxy[7]/img_h
                f.write(f"{int(cls)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")
        
        print(f"✅ Augmentation xyxyxyxy 라벨 저장: {xyxyxyxy_path}")
    
    
    def visualize(self, save_dir, img_name, info):
        """회전된 OBB 시각화"""
        img_name = img_name+".png"
        visualize_path = os.path.join(save_dir, img_name)
        image = copy.deepcopy(info['image'])
        corners = info['labels']
        for idx, corner in enumerate(corners):
            points = corner[1].reshape(4, 2).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(image, f"{idx}", tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
     
        cv2.imwrite(visualize_path, image)
        print(f"✅ OBB 시각화 저장: {visualize_path}")






