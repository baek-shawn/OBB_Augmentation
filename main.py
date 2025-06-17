from utils.pipeline import AugmentationPipeline
from utils.trial_manager import get_new_trial_path
from utils.loader import load_data, load_config
from augmentation.random_crop import RandomCrop
from augmentation.rotate import Rotate
from augmentation.scale import Scaling
from augmentation.translate import Translation
from augmentation.tile import Tiling
from utils.save_log import write_augment_log
import random
import os
import argparse

def main_pipeline(root_input, root_output, yaml_path,select_path, save_intermediate=False):
    
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
                
            elif method == "Scaling":
                augment = (method, Scaling(), hyps)
                augmentation_list.append(augment)
                
            elif method == "Translate":
                augment = (method, Translation(), hyps)
                augmentation_list.append(augment)
            
            elif method == "Tile":
                augment = (method, Tiling(), hyps)
                augmentation_list.append(augment)
                
                
        pipeline = AugmentationPipeline(
            transforms=augmentation_list,
            save_intermediate=save_intermediate
        )
        
        
        # Define Random ratio (Apply augmentation to image or not)
        if augment_info.get('RandomRatio', None) is None:
            random_ratio = 0
        else:
            random_ratio = augment_info["RandomRatio"].get('value', 0)
        
        random_num = round(random_ratio*10)
        random_list = [False] * random_num + [True] * (10-random_num)
        
        
        # If select_mode is enabled, retrieve the list of data to apply augmentation
        select_mode = augment_info['SelectMode'].get('value', 0) if augment_info.get('SelectMode', None) else None
        if select_mode is True:
            with open(select_path, "r", encoding="utf-8") as f:
                select_data = [line.strip() for line in f]
        
        

        # Apply pipeline
        idx = 0
        for data, image_name in load_data(image_input, label_input, new_root_output):
            random_choice = random.choice(random_list)
            if int(random_num)!=0 and random_choice:
                print(f"{idx}-{image_name}skip data!!!!!!!!!!!")
                idx+=1
                continue
            
            if select_mode is True:
                if image_name not in select_data:
                    print(f"{image_name} is not in select data, so skip data")
                    continue
            
            
            
            pipeline(data)
            
            break
        
        print(f"{new_root_output} works finish")
        print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True)
    parser.add_argument('--input', type=str, default='data')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--select_mode', action='store_true', help='If set, enables select mode')
    parser.add_argument('--select_path', type=str, default='select_data.txt')
    
    opt = parser.parse_args()
    
    main_pipeline(opt.input, opt.output, opt.yaml,opt.select_path)
    
  



   
        
    
