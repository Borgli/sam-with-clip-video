import argparse
import json
import platform
import re
import shutil
import signal
import sys

import psutil
import torch

from sam_with_clip import segment_video
import os
import subprocess
from pathlib import Path


def run_segment_video(log_folder_path):
    # Get GPU info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = {}
    if device.type == "cuda":
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory": torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
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

    with open(Path(log_folder_path).joinpath("system_info.json"), 'w') as f:
        json.dump(system_info, f)

    with open(Path(log_folder_path).joinpath("hyperparameters.json"), 'r') as f:
        hyperparameters = json.load(f)

    # Run script in another process called resource_monitor.py and send in the path of the log folder
    resource_monitor_process = subprocess.Popen([sys.executable, "resource_monitor.py", "--log_dir", str(log_folder_path)])

    segment_video(
        predicted_iou_threshold=hyperparameters["predicted_iou_threshold"],
        stability_score_threshold=hyperparameters["stability_score_threshold"],
        clip_threshold=hyperparameters["clip_threshold"],
        video_path="video.avi",
        query=hyperparameters["query"],
        output_path=os.path.join(log_folder_path, "output.avi")
    )

    # Stop the resource_monitor.py script
    os.kill(resource_monitor_process.pid, signal.SIGTERM)
    resource_monitor_process.wait()

    # Update the name of the log folder path and move files
    log_folder_path = Path(log_folder_path).rename(Path(
        f"{log_folder_path}_{gpu_info.get('name', platform.processor())}_{gpu_info.get('count', psutil.cpu_count())}_units"))

    print(f"Changed log folder name to '{log_folder_path}'")

    results = list(filter(lambda x: re.match(r"\d+-.*", x), os.listdir(log_folder_path)))
    search = re.search(r"(\d+)-(.*)", results[0])
    environment = search.group(2)
    log_folder_path = Path(log_folder_path).rename(f"{log_folder_path}-{environment}")

    jobid = search.group(1)
    pattern = re.compile(fr'{jobid}\.(err|out)$')
    files = [f for f in os.listdir(os.getcwd()) if pattern.search(f)]
    shutil.move(Path(files[0]).resolve(), Path(log_folder_path).joinpath(files[0]))
    shutil.move(Path(files[1]).resolve(), Path(log_folder_path).joinpath(files[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder_path")
    args = parser.parse_args()
    print(f"Started running main task for folder {args.log_folder_path}")
    run_segment_video(args.log_folder_path)
