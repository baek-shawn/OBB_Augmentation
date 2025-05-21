import csv
import os
from datetime import datetime


def write_augment_log(augment_info, save_dir):
    log_path = os.path.join(save_dir, "readme.txt")
    with open(log_path, "w") as f:
        f.write(f"Experiment Timestamp : {datetime.now()}\n")
        f.write(f"{'-'*40}\n")
        for method, hyps in augment_info.items():
            f.write(f"{method}   : {hyps}\n")
        f.write(f"{'-'*40}\n")
        
        