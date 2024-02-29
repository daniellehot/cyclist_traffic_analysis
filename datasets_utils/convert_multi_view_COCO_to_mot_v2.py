import os
import shutil
import argparse
from pycocotools.coco import COCO
import pandas as pd
import copy
import numpy as np

HEADER =  header = ["frame_number", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "confidence_score", "class_id", "visibility_score"]

# folder:number of images -> 0:998, 1000:1000, 2000:658, 2659:340, 3000:100
SPLITS_DEFINITION = {
    'train': ["1000", "2000", "2659"],
    'val': ["3000"],
    'test': ["0"]
}


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


# remove tracks with only a single occurrence
def remove_shadow_tracks(df):
    all_track_ids = df['track_id'].unique()
    for id in all_track_ids:
        rows_with_the_given_track_id = df.loc[df['track_id'] == id]
        print(f"track_id {id} occurences {rows_with_the_given_track_id.shape[0]}")
        if rows_with_the_given_track_id.shape[0] < 3:
            print(f"Dropping rows with track_id {id}")
            df = df.drop(rows_with_the_given_track_id.index)
    return df


def euclidean_distance(box1, box2):
    # Extracting coordinates from bounding boxes
    left1, top1, width1, height1 = box1
    left2, top2, width2, height2 = box2
    
    # Computing the center coordinates of each bounding box
    center1_x = left1 + width1 / 2
    center1_y = top1 + height1 / 2
    center2_x = left2 + width2 / 2
    center2_y = top2 + height2 / 2
    
    # Computing the Euclidean distance between the centers
    distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    return distance


def correct_object_ids(df):
    # get all registered IDs
    all_ids = df['track_id'].unique()
    print(f"Object ids before correction {all_ids}")
    corrected_annotations = []
    for id in all_ids:
        # get all tracks associated with the id
        tracks = df.loc[df['track_id'] == id]
        
        # sort tracks by a frame number
        tracks = tracks.sort_values(by=['frame_number'])
        tracks_np = tracks.to_numpy(dtype=np.float64) 

        # compute distance between bounding boxes in subsequent frames
        track_id = id
        for index in range(1, tracks_np.shape[0]):
            current_frame = tracks_np[index, :]
            current_bbox = current_frame[2:6]
            current_class = current_frame[7]
            previous_frame = tracks_np[index-1, :]
            previous_bbox = previous_frame[2:6]
            previous_class = previous_frame[7]
            distance = euclidean_distance(previous_bbox, current_bbox)
            if distance > 99 or current_class != previous_class:
                track_id += 0.1
            current_frame[1] = track_id

        corrected_annotations.append(tracks_np)

    corrected_annotations = np.vstack(corrected_annotations)
    df_corrected = pd.DataFrame(corrected_annotations, columns=HEADER)
    all_ids = df_corrected['track_id'].unique()
    print(f"Object ids after correction {all_ids}")
    return df_corrected


def order_annotations(df):
    first_detections = []
    all_ids = df['track_id'].unique()
    for id in all_ids:
        tracks = df.loc[df['track_id'] == id]
        first_detections.append(tracks.iloc[0]['frame_number'])
    
    combined = list(zip(first_detections, all_ids))
    sorted_combined = sorted(combined)
    ordered_list_of_track_ids = [x[1] for x in sorted_combined]
    return ordered_list_of_track_ids


# remap track_ids to be linearly increasing
def ensure_linear_track_ids(df, order):
    #all_track_ids = df['track_id'].unique()
    new_ids = np.arange(1, len(order)+1)
    
    # Map old track_ids to new ones
    id_map = {old_id: new_id for old_id, new_id in zip(order, new_ids)}
    print(id_map)
    df['track_id'] = df['track_id'].replace(id_map)

    return df


if __name__=="__main__":
    args = parse_args()
    input = args.input
    output = f"{args.output}/multi_view_mot"
    if os.path.exists(output):
        shutil.rmtree(output)
    
    train_folder = f"{output}/train" 
    os.makedirs(train_folder, exist_ok=True)
    test_folder = f"{output}/test"
    os.makedirs(test_folder, exist_ok=True)
    
    #data_output_path = f"{output}/data"
    #os.makedirs(data_output_path, exist_ok=True)
    
    # TODO Add drone data
    infrastructure_coco = coco = COCO(f"{input}/Sequence3-png/Infrastructure/infrastructure-mscoco.json")
    image_ids_map = generate_image_ids_map(infrastructure_coco)

    #infrastructure_image_folders = sorted([folder for folder in os.listdir(f"{input}/Sequence3-png/Infrastructure") if not folder.endswith((".png", ".json"))])
    for split in SPLITS_DEFINITION.keys():
        if split != "test":
            data_output_path = train_folder
        else:
            data_output_path = test_folder    
        
        image_folders = SPLITS_DEFINITION[split]
        for folder in image_folders:
            print(f"Processing folder {input}/Sequence3-png/Infrastructure/{folder}")

            save_images_to = f"{data_output_path}/{folder}/img1"
            os.makedirs(save_images_to, exist_ok=True)

            images = [file for file in os.listdir(f"{input}/Sequence3-png/Infrastructure/{folder}") if file.endswith(".png")]
            images = sorted(images)
            annotations_mot = pd.DataFrame(columns=HEADER)
            for idx, img in enumerate(images):
                idx += 1 # frame indexing should start at 1
                shutil.copy(f"{input}/Sequence3-png/Infrastructure/{folder}/{img}", f"{save_images_to}/{str(idx).zfill(6)}.png" )
                try:
                    annotation_ids = coco.getAnnIds(imgIds=image_ids_map[img])
                    annotations_coco = coco.loadAnns(annotation_ids)
                except:
                    print(f"No annotations for image {img}")
                    os.remove(f"{save_images_to}/{str(idx).zfill(6)}.png")
                annotations_mot = pd.concat([annotations_mot, convert_coco_to_mot(annotations_coco, idx)], ignore_index=True)
        
            annotations_mot = correct_object_ids(annotations_mot)
            annotations_mot = remove_shadow_tracks(annotations_mot)
            ordered_ids = order_annotations(annotations_mot)
            annotations_mot = ensure_linear_track_ids(annotations_mot, ordered_ids)
            annotations_mot = annotations_mot.astype(int)
            annotations_mot = annotations_mot.sort_values(by=['track_id', 'frame_number'])

            save_gt_to = f"{data_output_path}/{folder}/gt"
            os.makedirs(save_gt_to, exist_ok=True)
            annotations_mot.to_csv(f"{save_gt_to}/gt.txt", sep=',', header=False, index=False)

            save_det_to = f"{data_output_path}/{folder}/det"
            os.makedirs(save_det_to, exist_ok=True)
            no_tracks_sorted_mot_annotations = copy.deepcopy(annotations_mot)
            no_tracks_sorted_mot_annotations['track_id'] = -1
            no_tracks_sorted_mot_annotations.to_csv(f"{save_det_to}/det.txt", sep=',', header=False, index=False)
    