import argparse
import json
import sys

from sam_with_clip import segment_video
import os
import subprocess
from pathlib import Path


def run_segment_video(log_folder_path):
    # Run script in another process called resource_monitor.py and send in the path of the log folder
    resource_monitor_process = subprocess.Popen([sys.executable, "resource_monitor.py", "--log_dir", str(log_folder_path)])

    with open(Path(log_folder_path).joinpath("hyperparameters.json")) as f:
        hyperparameters = json.load(f)

    segment_video(
        predicted_iou_threshold=hyperparameters["predicted_iou_threshold"],
        stability_score_threshold=hyperparameters["stability_score_threshold"],
        clip_threshold=hyperparameters["clip_threshold"],
        video_path="video.avi",
        query=hyperparameters["query"],
        output_path=os.path.join(log_folder_path, "output.avi")
    )

    # Stop the resource_monitor.py script
    resource_monitor_process.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder_path")
    run_segment_video(parser.log_folder_path)
