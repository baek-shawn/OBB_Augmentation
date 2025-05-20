from utils.pipeline import AugmentationPipeline
from utils.trial_manager import get_new_trial_path
from utils.loader import load_data
from augmentation.random_crop import RandomCrop
from augmentation.rotate import Rotate
import random
import os

def main_pipeline(root_input, root_output, augment_info, save_intermediate=False, random_ratio=0):
    
    
    assert 0 < random_ratio <= 1.0
    root_output = get_new_trial_path(root_output)   # Create a root save directory "trial_{idx}" to prevent overwriting previous results.
    image_input = os.path.join(root_input, "images")
    label_input = os.path.join(root_input, "labels")
    
    # create augment_list
    augmentation_list = []
    for method, hyps in augment_info.items():
        
        if method == "RandomCrop":
            augment = (method, RandomCrop(), hyps)
            augmentation_list.append(augment)
            
        elif method == "Rotate":
            augment = (method, Rotate(), hyps)
            augmentation_list.append(augment)
            
        else:
            KeyError(f"{method} is not implementation")
    
    
    pipeline = AugmentationPipeline(
        transforms=augmentation_list,
        save_intermediate=save_intermediate
    )
    
    random_num = round(random_ratio*10)
    random_list = [False] * random_num + [True] * (10-random_num)

    # Apply pipeline
    idx = 0
    for data in load_data(image_input, label_input, root_output):
        random_choice = random.choice(random_list)
        if int(random_num)!=0 and random_choice:
            print(f"{idx} skip data!!!!!!!!!!!")
            idx+=1
            continue
        
        pipeline(data)

    

if __name__ == "__main__":
    # 1. 경로 설정
    root_input = "data"
    root_output = "output"
    
    
    augment_info = {
        "RandomCrop" : {
            "scale" : [0.3,0.5,0.7]
        },
        "Rotate" : {
            "angle" : [-15, -45]
        }
        
    }
    
    main_pipeline(root_input, root_output, augment_info, save_intermediate=False, random_ratio=0.3)
   
        
    
