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

        yield data
