import csv
import json
import os
import urllib
import time
from functools import lru_cache
from pathlib import Path
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import clip
import cv2
import gradio as gr
import numpy as np
import PIL
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "SAM")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
MAX_WIDTH = MAX_HEIGHT = 1024
TOP_K_OBJ = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def load_mask_generator() -> SamAutomaticMaskGenerator:
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint):
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


@lru_cache
def load_clip(
        name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess


def adjust_image_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image


@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str]) -> torch.Tensor:
    model, preprocess = load_clip()
    preprocessed = preprocess(crop).unsqueeze(0).to(device)
    tokens = clip.tokenize(texts).to(device)
    logits_per_image, _ = model(preprocessed, tokens)
    similarity = logits_per_image.softmax(-1).cpu()
    return similarity[0, 0]


def crop_image(image: np.ndarray, mask: Dict[str, Any]) -> PIL.Image.Image:
    x, y, w, h = mask["bbox"]
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y: y + h, x: x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = PIL.Image.fromarray(crop)
    return crop


def get_texts(query: str) -> List[str]:
    return [f"a picture of {query}", "a picture of background"]


def filter_masks(
        image: np.ndarray,
        masks: List[Dict[str, Any]],
        predicted_iou_threshold: float,
        stability_score_threshold: float,
        query: str,
        clip_threshold: float,
) -> List[Dict[str, Any]]:
    filtered_masks: List[Dict[str, Any]] = []

    for mask in sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]:
        if (
                mask["predicted_iou"] < predicted_iou_threshold
                or mask["stability_score"] < stability_score_threshold
                or image.shape[:2] != mask["segmentation"].shape[:2]
                or query
                and get_score(crop_image(image, mask), get_texts(query)) < clip_threshold
        ):
            continue

        filtered_masks.append(mask)

    return filtered_masks


def draw_masks(
        image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.7
) -> np.ndarray:
    for mask in masks:
        color = [randint(127, 255) for _ in range(3)]

        # draw mask overlay
        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(
            np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    return image


def segment(
        predicted_iou_threshold: float,
        stability_score_threshold: float,
        clip_threshold: float,
        image_path: str,
        query: str,
) -> PIL.ImageFile.ImageFile:
    mask_generator = load_mask_generator()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reduce the size to save gpu memory
    image = adjust_image_size(image)
    masks = mask_generator.generate(image)
    masks = filter_masks(
        image,
        masks,
        predicted_iou_threshold,
        stability_score_threshold,
        query,
        clip_threshold,
    )
    image = draw_masks(image, masks)
    image = PIL.Image.fromarray(image)
    return image


def segment_video(
        predicted_iou_threshold: float,
        stability_score_threshold: float,
        clip_threshold: float,
        video_path: str,
        query: str,
        output_path: str
) -> None:
    mask_generator = load_mask_generator()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video's frame size and fps
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Write to log
    with open(Path(output_path).parent.joinpath("video_info.json"), "w") as f:
        json.dump({
            "name": Path(video_path).name,
            "width": frame_width,
            "height": frame_height,
            "fps": fps,
            "total_frames": total_frames
        }, f)

    # Create a VideoWriter object to save the output video
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    with open(Path(output_path).parent.joinpath("runtime.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Frame", "Time Start", "Time End", "Performance"])
        frame_count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                print(f"Working on frame {frame_count} / {int(total_frames)}")

                time_now = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # reduce the size to save gpu memory
                frame = adjust_image_size(frame)
                masks = mask_generator.generate(frame)
                masks = filter_masks(
                    frame,
                    masks,
                    predicted_iou_threshold,
                    stability_score_threshold,
                    query,
                    clip_threshold,
                )
                frame = draw_masks(frame, masks)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Write the frame into the output video file
                out.write(frame)
                time_after = time.time()
                performance = time_after - time_now
                print(
                    f"This took {performance:.2f} seconds. ETA: {(total_frames - frame_count) * performance:.2f} seconds.")
                writer.writerow([frame_count, time_now, time_after, performance])
                frame_count += 1
            else:
                break

    # Release everything when the job is finished
    cap.release()
    out.release()
