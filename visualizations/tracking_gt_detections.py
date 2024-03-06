import pandas as pd
import cv2
import argparse
import os
import numpy as np

HEADER = ["frame_number", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "confidence_score", "class_id", "visibility_score"]

def parse_args():
    parser = argparse.ArgumentParser("Visualizations")
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--tracks", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def draw_text(image, text, bbox, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    # Calculate text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size[0], text_size[1]
    # Calculate the position to put text (in the middle of the bounding box)
    text_x = bbox[0] + (bbox[2] - bbox[0]) // 2 - text_width // 2
    text_y = bbox[1] + (bbox[3] - bbox[1]) // 2 + text_height // 2
    # Put text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness)


def draw_tracks(filepath, tracks, tracks_type=None):
    img = cv2.imread(filepath)
    frame_id = int(os.path.basename(filepath).split(".")[0])
    frame_tracks = tracks[tracks[0]==frame_id].to_numpy()
    frame_tracks = frame_tracks.astype(int)
    
    track_ids = frame_tracks[:, 1]
    left = frame_tracks[:, 2]
    top = frame_tracks[:, 3]
    right = frame_tracks[:, 2]+frame_tracks[:, 4]
    bottom = frame_tracks[:, 3]+frame_tracks[:, 5]
    
    if tracks_type == "gt":
        color = (0, 0, 255) 
    else:
        color = (0, 255, 0)
        
    thickness = 2
    for y1, x1, y2, x2, id in zip(left, top, right, bottom, track_ids):
        cv2.rectangle(img, (y1, x1), (y2, x2), color, thickness)
        text = f"{id}"
        draw_text(img, text, [y1, x1, y2, x2], color)

    cv2.imshow("test", img)
    cv2.waitKey(0)


def main(args):
    gt_file = os.path.join(args.seq, "gt", "gt.txt")
    gt = pd.read_csv(gt_file, header=None)
    image_folder = os.path.join(args.seq, "img1")
    images = [os.path.join(image_folder, f"{str(i+1).zfill(6)}.png") for i in range(len(os.listdir(image_folder)))]
    tracks = pd.read_csv(args.tracks, header=None)
    
    for image_path in images:
        gt_img = draw_tracks(image_path, gt, "gt")
        detections_img = draw_tracks(image_path, tracks, "detections")


if __name__=="__main__":
    main(parse_args())
