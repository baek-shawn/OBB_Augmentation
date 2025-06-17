import random
import numpy as np
import cv2
import os
import copy
import random
from .base import AugmentBase
import random


class Translation(AugmentBase):
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        # define hyperparameter
        translates = kwargs['ratio']  # 예: [(0.1, 0.2), (-0.2, 0.1)] (비율로 입력)
        padding = kwargs.get('padding', 0)

        # 선택된 translate 값
        if len(translates) > 1:
            translate = random.choice(translates)
        else:
            translate = translates[0]

        tx_frac, ty_frac = translate[0], translate[1]

        translated_info_dict = {}
        for idx, (img, oriented_bboxes, img_name) in enumerate(zip(self.image, self.oriented_bounding_boxes, self.image_name)):
            origin_img = img.copy()
            origin_img_h, origin_img_w = origin_img.shape[:2]
            obbs = copy.deepcopy(oriented_bboxes)

            # 실제 pixel 단위로 변환
            tx = tx_frac * origin_img_w
            ty = ty_frac * origin_img_h

            # Affine matrix
            translate_matrix = self.get_translation_affine_matrix(tx, ty)

            
            trans_img = cv2.warpAffine(origin_img, translate_matrix, (origin_img_w, origin_img_h), flags=cv2.INTER_LINEAR)

            
            trans_xywha, trans_xyxyxyxy = self.translate_obb(obbs, trans_img, translate_matrix)

            if img_name not in translated_info_dict:
                translated_info_dict[img_name] = {
                    "image": trans_img,
                    "xyxyxyxy": trans_xyxyxyxy,
                    "xywha": trans_xywha
                }

        return translated_info_dict

    def get_translation_affine_matrix(self, tx, ty):
        # 단순 이동 행렬
        affine_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        return affine_matrix

    def translate_obb(self, obbs, trans_img, trans_matrix):
        translated_xywha = []
        translated_xyxyxyxy = []

        img_h, img_w = trans_img.shape[:2]

        for idx, obb in enumerate(obbs):
            if len(obb) == 6:
                cls, x, y, w, h, a = obb
                corners = self.xywha_xyxyxyxy(x, y, w, h, a)
            elif len(obb) == 2:
                cls = obb[0]
                corners = obb[1].reshape(4, 2)
            else:
                print("OBB 형식이 맞지 않습니다.")
                return None

            trans_corners = self._translate_corners(corners, trans_matrix)

            intersection_area_ratio, intersection_coords, _ = self.calculate_intersection_area(trans_corners, 0, 0, img_w, img_h)

            if self._should_skip_box(intersection_area_ratio):
                continue

            if intersection_area_ratio < 0.999:
                trans_corners = self.fix_obb_with_min_area_rect(intersection_coords)
                if trans_corners is None:
                    continue

            trans_corners = trans_corners.flatten()
            translated_xyxyxyxy.append((cls, trans_corners))

        return translated_xywha, translated_xyxyxyxy

    def _translate_corners(self, corners, trans_matrix):
        ones = np.ones((4, 1))
        homogeneous = np.hstack([corners, ones]).T  # (3, 4)
        return np.dot(trans_matrix, homogeneous).T[:, :2]  # (4, 2)
