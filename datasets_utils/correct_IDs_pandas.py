import os
import shutil
import argparse
from pycocotools.coco import COCO
import pandas as pd
import copy

HEADER =  header = ["frame_number", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "confidence_score", "class_id", "visibility_score"]

def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="Annotation file")
    parser.add_argument("-o", "--output", type=str, help="Where to save mot multi-view-dataset")
    return parser.parse_args()


def generate_image_ids_map(coco):
    # Get all image IDs
    image_ids = coco.getImgIds()
    # Assosicate image IDs with images in the sequence
    image_id_map = {}
    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]         
        image_id_map[image_info['file_name'].split("/")[-1]] = image_id
    return image_id_map


def convert_coco_to_mot(coco_annotations, frame_number):
    #<frame_number>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence_score>, <class_id>, <visibility_score>
    mot_annotations = pd.DataFrame(columns=HEADER)
    for idx, ann in enumerate(coco_annotations):
        mot_annotations.loc[idx, "frame_number"] = frame_number
        mot_annotations.loc[idx, "track_id"] = ann['object_id']
        mot_annotations.loc[idx, "bb_left"] = ann['bbox'][0]
        mot_annotations.loc[idx, "bb_top"] = ann['bbox'][1]
        mot_annotations.loc[idx, "bb_width"] = ann['bbox'][2]
        mot_annotations.loc[idx, "bb_height"] = ann['bbox'][3]
        mot_annotations.loc[idx, "confidence_score"] = 1
        mot_annotations.loc[idx, "class_id"] = ann['category_id']
        mot_annotations.loc[idx, "visibility_score"] = 1
    return mot_annotations


# remove tracks with only a single occurence
def remove_shadow_tracks(df):
    all_track_ids = df['track_id'].unique()
    for id in all_track_ids:
        rows_with_the_given_track_id = df.loc[df['track_id'] == id]
        print(f"track_id {id} occurences {rows_with_the_given_track_id.shape[0]}")
        if rows_with_the_given_track_id.shape[0] == 1:
            print(f"Dropping rows with track_id {id}")
            df = df.drop(rows_with_the_given_track_id.index)
    return df


# remap track_ids to be linearly increasing
def ensure_linear_track_ids(df):
    all_track_ids = df['track_id'].unique()
    print(f"all_track_ids before correction {all_track_ids}")

    linearly_increasing = False
    while not linearly_increasing:
        all_track_ids = df['track_id'].unique()
        print(f"all_track_ids correction {all_track_ids}")
        for i in range(1, len(all_track_ids)):
            if all_track_ids[i] == all_track_ids[i-1] + 1:
                linearly_increasing = True
            if all_track_ids[i] != all_track_ids[i-1] + 1:
                df['track_id'] = df['track_id'].replace(all_track_ids[i], all_track_ids[i-1] + 1)
                linearly_increasing = False
                break #break the for loop and start again; this repeats untill all track_ids are remapped to be linearly increasing

    all_track_ids = df['track_id'].unique()
    print(f"all_track_ids after correction {all_track_ids}")
    return df


def ensure_linear_frames(df):
    all_track_ids = df['track_id'].unique()
    for id in all_track_ids:
        all_frames = df[df['track_id'] == id]['frame_number'].tolist()
        
        track_breaks = [0]
        for i in range(1, len(all_frames)):
            if all_frames[i] != all_frames[i-1] + 1:
                track_breaks.append(i)

        new_tracks = []
        for i in range(1, len(track_breaks)):
            new_tracks.append(all_frames[track_breaks[i-1]:track_breaks[i]])
    
        for idx, track in enumerate(new_tracks):
            new_id = id +idx/10 
            for frame in track:
                df.loc[(df['frame_number'] == frame) & (df['track_id'] == id), 'track_id'] = new_id 
    return df
                           

    

if __name__=="__main__":
    args = parse_args()
    input = args.input
    
    infrastructure_coco = coco = COCO(f"{input}/Sequence3-png/Infrastructure/infrastructure-mscoco.json")
    image_ids_map = generate_image_ids_map(infrastructure_coco)

    infrastructure_image_folders = sorted([folder for folder in os.listdir(f"{input}/Sequence3-png/Infrastructure") if not folder.endswith((".png", ".json"))])
    for folder in infrastructure_image_folders:
        print(f"Processing folder {input}/Sequence3-png/Infrastructure/{folder}")

        images = [file for file in os.listdir(f"{input}/Sequence3-png/Infrastructure/{folder}") if file.endswith(".png")]
        images = sorted(images)
        annotations_mot = pd.DataFrame(columns=HEADER)
        for idx, img in enumerate(images):
            idx += 1 # frame indexing should start at 1
            try:
                annotation_ids = coco.getAnnIds(imgIds=image_ids_map[img])
                annotations_coco = coco.loadAnns(annotation_ids)
                #for ann in annotations_coco:
                #    all_object_ids[folder].append(ann['object_id'])
            except:
                print(f"No annotations for image {img}")
            annotations_mot = pd.concat([annotations_mot, convert_coco_to_mot(annotations_coco, idx)], ignore_index=True)
        track_id_sorted_mot_annotations = annotations_mot.sort_values(by=['track_id', 'frame_number'])
        track_id_sorted_mot_annotations['track_id'] = track_id_sorted_mot_annotations['track_id'] - track_id_sorted_mot_annotations['track_id'].min() + 1 
        #track_id_sorted_mot_annotations.to_csv(f"./gt_{folder}.txt", sep=',', header=False, index=False)
        # remove tracks with only a single annotation
        track_id_sorted_mot_annotations = remove_shadow_tracks(track_id_sorted_mot_annotations)
        # confirm frames increase linearly for a given track, i.e., remove duplicate use of IDs
        track_id_sorted_mot_annotations = ensure_linear_frames(track_id_sorted_mot_annotations)
        # confirm track ids increase linearly
        track_id_sorted_mot_annotations = ensure_linear_track_ids(track_id_sorted_mot_annotations)
        #print(track_id_sorted_mot_annotations)
        print("\n")