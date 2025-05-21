from utils.pipeline import AugmentationPipeline
from utils.trial_manager import get_new_trial_path
from utils.loader import load_data, load_config
from augmentation.random_crop import RandomCrop
from augmentation.rotate import Rotate
from utils.save_log import write_augment_log
import random
import os

def main_pipeline(root_input, root_output, yaml_path, save_intermediate=False):
    
    augment_infos = load_config(yaml_path)  # load augmentation info
    
    image_input = os.path.join(root_input, "images")
    label_input = os.path.join(root_input, "labels")
    
    for augment_info in augment_infos:
        new_root_output = get_new_trial_path(root_output)   # Create a root save directory "trial_{idx}" to prevent overwriting previous results.
        write_augment_log(augment_info, new_root_output)    # Write augmentation info
        # create augment_list
        augmentation_list = []
        for method, hyps in augment_info.items():
            if method == "RandomCrop":
                augment = (method, RandomCrop(), hyps)
                augmentation_list.append(augment)
                
            elif method == "Rotate":
                augment = (method, Rotate(), hyps)
                augmentation_list.append(augment)
                
        pipeline = AugmentationPipeline(
            transforms=augmentation_list,
            save_intermediate=save_intermediate
        )
        
        
        # Define Random ratio (Apply augmentation to image or not)
        if augment_info['RandomRatio'] is None:
            random_ratio = 0
        else:
            random_ratio = augment_info["RandomRatio"].get('value', 0)
        
        random_num = round(random_ratio*10)
        random_list = [False] * random_num + [True] * (10-random_num)

        # Apply pipeline
        idx = 0
        for data, image_name in load_data(image_input, label_input, new_root_output):
            random_choice = random.choice(random_list)
            if int(random_num)!=0 and random_choice:
                print(f"{idx}-{image_name}skip data!!!!!!!!!!!")
                idx+=1
                continue
            
            pipeline(data)
        
        print(f"{new_root_output} works finish")
        print("==============================")

    

if __name__ == "__main__":
    # 1. 경로 설정
    root_input = "data"
    root_output = "output"
    yaml_path = "augment.yaml"
    

    main_pipeline(root_input, root_output, yaml_path, save_intermediate=False)
   
        
    
