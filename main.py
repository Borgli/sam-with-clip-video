from sam_with_clip import segment_video

import torch.multiprocessing as mp


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
    segment_video(
        predicted_iou_threshold=0.9,
        stability_score_threshold=0.8,
        clip_threshold=0.9,
        video_path="video.avi",
        query="polyp",
        output_path="output.avi"
    )


if __name__ == '__main__':
    run_segment_video()
