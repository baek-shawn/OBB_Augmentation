from utils.pipeline import AugmentationPipeline
from utils.trial_manager import get_new_trial_path
from utils.loader import load_data
from augmentation.random_crop import RandomCrop
from augmentation.rotate import Rotate

import os

if __name__ == "__main__":
    # 1. 경로 설정
    root_input = "data"
    root_output = "output"
    root_output = get_new_trial_path(root_output)
    
    image_input = os.path.join(root_input, "images")
    label_input = os.path.join(root_input, "labels")

    # 2. 유저가 설정하는 부분
    augmentations_list = [
        ("RandomCrop", RandomCrop(), {"scale": 0.7}),
        ("Rotate", Rotate(), {"angle": 15}),
    ]
    
    save_intermediate = False

    # 3. 파이프라인 생성
    pipeline = AugmentationPipeline(
        transforms=augmentations_list,
        save_intermediate=save_intermediate
    )
    
    # 4. 데이터 불러와서 파이프라인 적용
    for data in load_data(image_input, label_input, root_output):
        pipeline(data)
