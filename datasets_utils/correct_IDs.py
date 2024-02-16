from pycocotools.coco import COCO
import argparse
import cv2
import os, shutil
import pandas as pd
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="Annotation file")
    parser.add_argument("--seq", type=str, help="Sequence to visualize")
    parser.add_argument("-o", "--output", type=str, help="Output folder")
    parser.add_argument("--png", action='store_true')
    parser.add_argument("--jpg", action='store_true')
    return parser.parse_args()


def generate_image_ids_map(coco, sequence):
    # Get all image IDs
    image_ids = coco.getImgIds()

    # Assosicate image IDs with images in the sequence
    image_id_map = {}
    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]         
        for img in sequence:
            if img.split("/")[-1] == image_info['file_name'].split("/")[-1]:
                image_id_map[img] = image_id
    return image_id_map


def get_object_ids(annotations):
    object_ids = []
    for ann in annotations:
        object_ids.append(ann['object_id'])
    return object_ids


def compare_lists(list1, list2):
    # Find the length of the shorter list
    min_length = min(len(list1), len(list2))
    
    # Use list comprehension to find elements that are different
    different_values = [value1 for value1, value2 in zip(list1[:min_length], list2[:min_length]) if value1 != value2]
    
    # Append remaining elements from the longer list, if any
    different_values.extend(list1[min_length:])
    different_values.extend(list2[min_length:])
    
    return different_values


def get_duplicate_ids(object_ids):
    frame_previous = []
    frame_now = []
    frame_id = 0

    removed_ids = []
    added_ids = []
    frames_to_fix = {}
    ids_to_fix = {}

    for k,v in object_ids.items():
        if frame_id == 0:
            frame_now = v
        else:
            frame_previous = frame_now
            frame_now = v
            frames_to_fix[frame_id] = []

            if len(frame_now) == len(frame_previous):
                pass

            elif len(frame_now) > len(frame_previous):
                # What IDs were added
                for id in frame_now:
                    if id not in frame_previous:
                        added_ids.append(id)
                    if id not in frame_previous and id in removed_ids:
                        idx

            elif len(frame_now) < len(frame_previous):
                # What IDs have already been removed, NO REPETITION 
                for id in frame_previous:
                    if id not in frame_now and id not in removed_ids:
                        removed_ids.append(id)
                
            #print(f"Frame {frame_id} IDs_now {frame_now} IDs_before{frame_previous} Added {added_ids} Removed {removed_ids}")    

        frame_id += 1   
        #print(f"Frame {frame_id} IDs_now {frame_now} IDS_before{frame_previous}")
    for k,v in frames_to_fix.items():
        if len(v) != 0:
            print(frames_to_fix)
            

def main(args):
    if args.jpg:
        sequence = sorted([img for img in os.listdir(args.seq) if img.endswith(".jpg")])
    if args.png:
        sequence = sorted([img for img in os.listdir(args.seq) if img.endswith(".png")])

    coco = COCO(args.input)
    image_ids_map = generate_image_ids_map(coco, sequence) 

    object_ids = {}
    for idx, img in enumerate(sequence):
        if idx == 0:
            continue
        try:
            annotation_ids = coco.getAnnIds(imgIds=image_ids_map[img])
            annotations = coco.loadAnns(annotation_ids)
            object_ids[img] = get_object_ids(annotations)
        except:
            print(f"No annotations for image {img}")

    get_duplicate_ids(object_ids)



if __name__=="__main__":
    main(parse_args())