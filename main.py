import sys

from sam_with_clip import segment_video

import psutil
import torch.multiprocessing as mp
import os
import subprocess
import platform
import datetime
import torch
from pathlib import Path


def run_benchmark():
    mp.set_start_method("spawn", force=True)

    p = mp.Process(target=segment_video,
                   kwargs={
                       "predicted_iou_threshold": 0.9,
                       "stability_score_threshold": 0.8,
                       "clip_threshold": 0.9,
                       "video_path": "video.avi",
                       "query": "polyp",
                       "output_path": "output.avi"
                   }
                   )

    p.start()
    print("Started")
    p.join()
    print("Finished")


def run_segment_video():
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

    # Get system info
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": gpu_info
    }
    print(f"System info: {system_info}")

    # Create log folder with the name of the experiment ID and include the GPU model name
    log_folder_path = Path(f"{exp_id}_{gpu_info.get('name', 'cpu')}").resolve()
    print(f"Creating log folders at {log_folder_path}")
    log_folder_path.mkdir(parents=True, exist_ok=True)

    # Run script in another process called resource_monitor.py and send in the path of the log folder
    resource_monitor_process = subprocess.Popen([sys.executable, "resource_monitor.py", "--log_dir", str(log_folder_path)])

    segment_video(
        predicted_iou_threshold=0.9,
        stability_score_threshold=0.8,
        clip_threshold=0.9,
        video_path="video.avi",
        query="polyp",
        output_path="output.avi"
    )

    # Stop the resource_monitor.py script
    resource_monitor_process.terminate()


if __name__ == '__main__':
    run_segment_video()
