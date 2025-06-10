import os
import cv2
import numpy as np

def load_data(image_input, label_input, save_dir):
    for image_file in os.listdir(image_input):
        if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(image_input, image_file)
        label_path = os.path.join(label_input, os.path.splitext(image_file)[0] + ".txt")
        image_name = os.path.splitext(image_file)[0]
        
        if not os.path.exists(label_path):
            print(f"❌ 라벨 파일 없음: {label_path}, 스킵합니다.")
            continue
        
        image = cv2.imread(image_path)
        obbs = []

        with open(label_path, "r") as f:
            for line in f:
                if line.strip().split()[0] == "YOLO_OBB":
                    continue
                values = list(map(float, line.strip().split()))
                cls, x, y, w, h, a = values
                obbs.append((cls, x, y, w, h, a))

        obbs = np.array(obbs)

        data = {
            "image": image,
            "labels": obbs,
            "image_name": image_name,
            "save_dirs": save_dir,
        }

        yield data, image_name



from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Union
import yaml

@dataclass
class RandomCropParams:
    scale: List[float]

@dataclass
class RotateParams:
    angle: List[int]
    
@dataclass
class ScaleParams:
    scale: List[List]

@dataclass
class TranslateParams:
    ratio: List[List]

@dataclass
class RandomRatioParams:
    value: float

@dataclass
class SelectModeParams:
    value: bool

# @dataclass
# class AugmentConfig:
#     RandomCrop: Optional[RandomCropParams] = None
#     Rotate: Optional[RotateParams] = None
#     Scaling: Optional[ScaleParams] = None
#     Translate: Optional[TranslateParams] = None
#     RandomRatio: Optional[RandomRatioParams] = None
#     SelectMode : Optional[SelectModeParams] = None

# @dataclass
# class Config:
#     augment_info: List[AugmentConfig]


def load_config(yaml_path: str) -> List[dict]:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    augment_info = []
    for aug_dict in data['augment_info']:
        kwargs = {}
        for aug_name, params in aug_dict.items():
            if aug_name == 'RandomCrop':
                kwargs['RandomCrop'] = asdict(RandomCropParams(**aug_dict['RandomCrop']))
            elif aug_name == 'Rotate':
                kwargs['Rotate'] = asdict(RotateParams(**aug_dict['Rotate']))
            elif aug_name == 'Scaling':
                kwargs['Scaling'] = asdict(ScaleParams(**aug_dict['Scaling']))  
            elif aug_name == 'Translate':
                kwargs['Translate'] = asdict(TranslateParams(**aug_dict['Translate']))
            elif aug_name == 'RandomRatio':
                kwargs['RandomRatio'] = asdict(RandomRatioParams(**aug_dict['RandomRatio']))
            elif aug_name == 'SelectMode':
                kwargs['SelectMode'] = asdict(SelectModeParams(**aug_dict['SelectMode']))
            else:
                raise ValueError(f"Unknown augmentation: {aug_name}")
       
        # dataclass → dict 변환
        # augment_info.append(asdict(AugmentConfig(**kwargs)))
        augment_info.append(kwargs)
    
    
    return augment_info