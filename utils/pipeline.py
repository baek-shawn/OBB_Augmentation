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
            
            result = transform(**hyps)
            
            key = list(result.keys())[0]
            data["image"] = result[key]["image"]
            data["labels"] = result[key]["xyxyxyxy"]
            data["image_name"] = key

            history.append(name)

            if self.save_intermediate and idx < transform_len - 1:
                self._save(data, history, transform, mode="intermediate")

        self._save(data, history, transform, mode="final")
        return data

    def _save(self, data, history, transform, mode='final', visualize=True):
        
        idx = random.randint(0, 100000) 
        postfix = "_".join(history)
        save_image_name = f"{data['image_name']}_{postfix}_{idx}"
        
        img_save_dir = os.path.join(data["save_dirs"], mode, "images")
        xywhr_save_dir = os.path.join(data["save_dirs"], mode, "xywhr")
        xyxyxyxy_save_dir = os.path.join(data["save_dirs"], mode, "xyxyxyxy")
        visualize_save_dir = os.path.join(data["save_dirs"], mode, "visualize")
        
        for folder in [img_save_dir, xywhr_save_dir, xyxyxyxy_save_dir, visualize_save_dir]:
            os.makedirs(folder, exist_ok=True)
        
        transform.save_img(img_save_dir, save_image_name, data)
        transform.save_xyxyxyxy(xyxyxyxy_save_dir, save_image_name, data, data['image'].shape[1], data['image'].shape[0])
        if visualize:
            transform.visualize(visualize_save_dir, save_image_name, data)
