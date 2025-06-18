import os
import random

class AugmentationPipeline:
    def __init__(self, transforms, save_intermediate=False):
        self.transforms = transforms
        self.save_intermediate = save_intermediate

    def __call__(self, data):
        history = []
        transform_len = len(self.transforms)
        
        for idx, (name, transform, hyps) in enumerate(self.transforms):
            transform.set_data(data)
    
            if name == "RandomCrop":
                if hyps is None:
                    continue
                hyp = hyps.get('scale', None)
                if hyp is None:
                    raise ValueError(f"'scale' parameter must be specified for {name} augmentation.")
            elif name == "Rotate":
                if hyps is None:
                    continue
                hyp = hyps.get('angle', None)
                if hyp is None:
                    raise ValueError(f"'angle' parameter must be specified for {name} augmentation.")
            elif name == "Scaling":
                if hyps is None:
                    continue
                hyp = hyps.get('scale', None)
                if hyp is None:
                    raise ValueError(f"'scale' parameter must be specified for {name} augmentation.")
            elif name == "Translate":
                if hyps is None:
                    continue
                hyp = hyps.get('ratio', None)
                if hyp is None:
                    raise ValueError(f"'ratio' parameter must be specified for {name} augmentation.")
            elif name == "Tile":
                if hyps is None:
                    continue
                hyp = hyps.get('scale', None)
                if hyp is None:
                    raise ValueError(f"'scale' parameter must be specified for {name} augmentation.")
            elif name == "MultipleScale":
                
                if hyps is None:
                    continue
                hyp1 = hyps.get('px_value', None)
                hyp2 = hyps.get('ratio_value', None)
                if hyp1 is None and hyp2 is None: 
                    raise ValueError(f"'px_value' or 'ratio_value' parameter must be specified for {name} augmentation.")
                
            result = transform(**hyps)
            
            # overlap the results
            img_names = list(result.keys())
            img_list = []
            labels = []
            image_name_list = []
            for img_name in img_names:
                img_list.append(result[img_name]['image'])
                labels.append(result[img_name]['xyxyxyxy'])
                image_name_list.append(img_name)
            data["image"] = img_list
            data["labels"] = labels
            data["image_name"] = image_name_list
           

            history.append(name)

            if self.save_intermediate and idx < transform_len - 1:
                self._save(data, history, transform, mode="intermediate")
        
        self._save(data, history, transform, mode="final")
        
        return data

    def _save(self, data, history, transform, mode='final', visualize=True):
        
        
        img_save_dir = os.path.join(data["save_dirs"], mode, "images")
        xywhr_save_dir = os.path.join(data["save_dirs"], mode, "xywhr")
        xyxyxyxy_save_dir = os.path.join(data["save_dirs"], mode, "xyxyxyxy")
        visualize_save_dir = os.path.join(data["save_dirs"], mode, "visualize")
        
        for folder in [img_save_dir, xywhr_save_dir, xyxyxyxy_save_dir, visualize_save_dir]:
            os.makedirs(folder, exist_ok=True)
        
        for img, label, img_name in zip(data['image'], data['labels'], data['image_name']):
        
            idx = random.randint(0, 100000) 
            postfix = "_".join(history)
            save_image_name = f"{img_name}_{postfix}_{idx}"
            transform.save_img(img_save_dir, save_image_name, img)
            transform.save_xyxyxyxy(xyxyxyxy_save_dir, save_image_name, label, img.shape[1], img.shape[0])
            if visualize:
                transform.visualize(visualize_save_dir, save_image_name, img, label)
