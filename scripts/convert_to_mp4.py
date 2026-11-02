import re
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def natural_sort_key(filename):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


def group_images_by_prefix(folder):
    groups = {}
    for file in os.listdir(folder):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # identify prefix up to the first digit
        m = re.match(r"([^\d]+)", file)
        if not m:
            continue

        prefix = m.group(1)

        groups.setdefault(prefix, []).append(file)

    return groups


def images_to_video(image_paths, output_path, fps=30):
    first_frame = cv2.imread(image_paths[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in tqdm(image_paths, desc=f"Writing {output_path}", unit="frame"):
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Saved video: {output_path}")


def convert_folder_to_mp4(folder_path, fps=30):
    folder = Path(folder_path)
    groups = group_images_by_prefix(folder)

    for prefix, files in groups.items():
        files = sorted(files, key=natural_sort_key)
        full_paths = [str(folder / f) for f in files]
        output_file = folder / f"{prefix.rstrip('_')}.mp4"
        images_to_video(full_paths, str(output_file), fps=fps)

        print(f"{prefix}: {len(files)} images -> {output_file}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "eval_results"
    convert_folder_to_mp4(folder, fps=30)
