import random
import numpy as np
import cv2
import os
import copy
import random
from .base import AugmentBase
import random



class Resize(AugmentBase):
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        fixed_value = kwargs["px_value"]
        resize_value = kwargs['ratio_value']
        resized_info_dict = {}
    
        for idx, (img, oriented_bboxes, img_name) in enumerate(zip(self.image, self.oriented_bounding_boxes, self.image_name)):
            h, w = img.shape[:2]
            # First choose fixed value, next to ratio
            if fixed_value:
                if len(fixed_value) > 1:
                    choice_value = random.choice(fixed_value)
                else:
                    choice_value = fixed_value[0]
                new_w, new_h = choice_value
                scale_x, scale_y = new_w / w, new_h / h
            elif resize_value:
                if len(resize_value) > 1:
                    choice_value = random.choice(resize_value)
                else:
                    choice_value = resize_value[0]
                scale_x, scale_y = choice_value
                new_w, new_h = int(w * scale_x), int(h * scale_y)
            else:
                raise ValueError("resize_mode must be either 'px_value' or 'ratio_value'.")

            # 1. 이미지는 OpenCV로 바로 리사이즈
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 2. 바운딩 박스는 Affine 행렬로 좌표 변환
            affine_matrix = np.array([[scale_x, 0, 0],
                                      [0, scale_y, 0]], dtype=np.float32)

            resized_xywha, resized_xyxyxyxy = self.resize_obb(oriented_bboxes, resized_img, affine_matrix)

            resized_info_dict[img_name] = {
                "image": resized_img,
                "xyxyxyxy": resized_xyxyxyxy,
                "xywha": resized_xywha
            }

        return resized_info_dict

    def resize_obb(self, obbs, resized_img, affine_matrix):
        resized_xywha = []
        resized_xyxyxyxy = []
        img_h, img_w = resized_img.shape[:2]

        for obb in obbs:
            if len(obb) == 6:
                cls, x, y, w, h, a = obb
                corners = self.xywha_xyxyxyxy(x, y, w, h, a)
            elif len(obb) == 2:
                cls = obb[0]
                corners = obb[1].reshape(4, 2)
            else:
                continue

            ones = np.ones((4, 1))
            homo_corners = np.hstack([corners, ones]).T  # (3, 4)
            resized_corners = np.dot(affine_matrix, homo_corners).T

            area_ratio, intersection_coords, _ = self.calculate_intersection_area(
                resized_corners, 0, 0, img_w, img_h)

            if self._should_skip_box(area_ratio):
                continue

            if area_ratio < 1 - 1e-6:
                resized_corners = self.fix_obb_with_min_area_rect(intersection_coords)
                if resized_corners is None:
                    continue

            resized_corners = resized_corners.flatten()
            resized_xyxyxyxy.append((cls, resized_corners))

        return resized_xywha, resized_xyxyxyxy