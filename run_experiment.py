import json
import os
import re
import shutil
import subprocess
import datetime
from pathlib import Path

if __name__ == '__main__':
    # Generate experiment ID from current time
    exp_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(f"Starting experiment: {exp_id}")

    hyperparameters = {
        "predicted_iou_threshold": 0.9,
        "stability_score_threshold": 0.8,
        "clip_threshold": 0.9,
        "query": "polyp"
    }

    # Create log folder with the name of the experiment ID and include the GPU model name
    log_folder_path = Path(f"{exp_id}").resolve()
    print(f"Creating log folders at '{log_folder_path}'")
    log_folder_path.mkdir(parents=True, exist_ok=True)

    with open(Path(log_folder_path).joinpath("hyperparameters.json"), 'w') as f:
        json.dump(hyperparameters, f)

    if Path("run_sam_with_clip.sh").exists():
        shutil.copy2(Path("run_sam_with_clip.sh"), Path(log_folder_path).joinpath("run_sam_with_clip.sh"))

    experiment = subprocess.Popen(
        ["sbatch", "--parsable", Path(log_folder_path).joinpath("run_sam_with_clip.sh"), log_folder_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = experiment.communicate()
    jobid = stdout.decode().strip()
    if jobid.isdigit():
        node = subprocess.Popen(
            ["squeue", "-h", "-o", '"%P"', f"-j{jobid}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        nodeout, nodeerr = node.communicate()
        info = nodeout.decode().strip().strip('"')
        print(f"Job {jobid} added to the queue on partition '{info}'. ")
        with open(Path(log_folder_path).joinpath(f"{jobid}-{info}"), "w") as f:
            f.write(f"{jobid}-{info}")

    if stderr.decode():
        print(stderr.decode())


