import json
import shutil
import subprocess
import datetime
import platform
from pathlib import Path

import psutil
import torch

if __name__ == '__main__':
    # Generate experiment ID from current time
    exp_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Get GPU info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = {}
    if device.type == "cuda":
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory": torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
        }

    print(f"Starting experiment: {exp_id}")

    hyperparameters = {
        "predicted_iou_threshold": 0.9,
        "stability_score_threshold": 0.8,
        "clip_threshold": 0.9,
        "query": "polyp"
    }

    # Get system info
    system_info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "num_cores": psutil.cpu_count(),
        "frequency": psutil.cpu_freq(),
        "gpu": gpu_info
    }
    print(f"System info: {system_info}")

    # Create log folder with the name of the experiment ID and include the GPU model name
    log_folder_path = Path(
        f"{exp_id}_{gpu_info.get('name', platform.processor())}_{gpu_info.get('count', psutil.cpu_count)}").resolve()
    print(f"Creating log folders at {log_folder_path}")
    log_folder_path.mkdir(parents=True, exist_ok=True)

    with open(Path(log_folder_path).joinpath("hyperparameters.json"), 'w') as f:
        json.dump(hyperparameters, f)

    with open(Path(log_folder_path).joinpath("system_info.json"), 'w') as f:
        json.dump(system_info, f)

    if Path("run_sam_with_clip.sh").exists():
        shutil.copy2(Path("run_sam_with_clip.sh"), Path(log_folder_path).joinpath("run_sam_with_clip.sh"))

    experiment = subprocess.Popen([Path(log_folder_path).joinpath("run_sam_with_clip.sh").resolve(), log_folder_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = experiment.communicate()


