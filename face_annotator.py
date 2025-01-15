#!/usr/bin/env python3
import argparse
import glob
import itertools
import os
from collections import deque
from pathlib import Path
from typing import Any, Tuple, Union

import cv2
import numpy as np
from lib.akari_yolo_lib.oakd_spatial_yolo import OakdSpatialYolo
from lib.akari_yolo_lib.util import download_file

save_num: int = 0


def detection_to_annotation(
    frame: np.ndarray, detection: Any, id: int) -> str:
    annotation_text = ""
    x1 = detection.xmin
    x2 = detection.xmax
    y1 = detection.ymin
    y2 = detection.ymax
    center_x: float = (x1 + x2) / 2
    center_y: float = (y1 + y2) / 2
    width: float = x2 - x1
    height: float = y2 - y1
    annotation_text += f"{id} {center_x} {center_y} {width} {height}\n"
    return annotation_text


def save_face_frame(
    image: np.ndarray,
    detection: any,
    path: str,
    name: str,
    id: int,
) -> bool:
    global save_num
    save_path: str = path + "/" + name
    image_path = save_path + ".jpg"
    cv2.imwrite(image_path, image)
    annotation: str = detection_to_annotation(image, detection, id)
    annotation_path = save_path + ".txt"
    with open(annotation_path, "w") as file:
        file.write(annotation)
    print(f"Saved. name: {name}")
    return True


def main() -> None:
    global save_num
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fps",
        help="Camera frame fps. This should be smaller than nn inference fps",
        default=5,
        type=int,
    )
    parser.add_argument("-n", "--name", type=str, required=True, help="person name")
    parser.add_argument("-i", "--id", type=str, required=True, help="person id")
    args = parser.parse_args()
    model_path = "model/human_parts.blob"
    config_path = "config/human_parts.json"
    download_file(
        model_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/human_parts/human_parts.blob",
    )
    download_file(
        config_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/human_parts/human_parts.json"
    )
    path = f"data/{args.name}/"
    os.makedirs(path, exist_ok=True)
    files = glob.glob(path + "/*")
    for file in files:
        file_name = Path(file).stem
        try:
            file_num = int(file_name.split('_')[-1])
            if file_num >= save_num:
                save_num = file_num + 1
        except BaseException:
            pass

    oakd_spatial_yolo = OakdSpatialYolo(
        config_path=config_path,
        model_path=model_path,
        fps=args.fps,
        track_targets=["face"],
    )
    saving = False
    while True:
        frame = None
        detections = []
        frame, detections = oakd_spatial_yolo.get_frame()
        raw_frame = oakd_spatial_yolo.get_raw_frame()
        if frame is not None:
            oakd_spatial_yolo.display_frame("nn", frame, detections)
            if len(detections) > 0 and saving:
                closest_face_num = 0
                closest_face_distance = None
                # 複数人の顔が認識された場合、一番近い人の顔を保存する
                for num, detection in enumerate(detections):
                    if closest_face_distance is None:
                        closest_face_distance = detection.spatialCoordinates.z
                    else:
                        if detection.spatialCoordinates.z < closest_face_distance:
                            closest_face_distance = detection.spatialCoordinates.z
                            closest_face_num = num
                name = f"{args.name}_{str(save_num).zfill(3)}"
                print(f"closest_face_num: {closest_face_num}")
                save_face_frame(
                    raw_frame,
                    detections[closest_face_num],
                    path=path,
                    name=name,
                    id=args.id,
                )
                save_num += 1
        key = cv2.waitKey(10)
        if key == ord("s"):
            saving = not saving
        elif key == ord("q"):
            break


if __name__ == "__main__":
    main()
