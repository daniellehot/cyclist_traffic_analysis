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


if __name__=="__main__":
    args = parse_args()
    input = args.input
    output = f"{args.output}/multi_view_mot"

    #os.makedirs(f"{output}/multi_view_mot_dataset/train", exist_ok=True)
    #os.makedirs(f"{output}/multi_view_mot_dataset/test", exist_ok=True)
    if os.path.exists(output):
        shutil.rmtree(output)
    data_output_path = f"{output}/data"
    os.makedirs(data_output_path, exist_ok=True)

    infrastructure_coco = coco = COCO(f"{input}/Sequence3-png/Infrastructure/infrastructure-mscoco.json")
    image_ids_map = generate_image_ids_map(infrastructure_coco)

    #all_object_ids = {folder:[] for folder in infrastructure_image_folders}
    infrastructure_image_folders = sorted([folder for folder in os.listdir(f"{input}/Sequence3-png/Infrastructure") if not folder.endswith((".png", ".json"))])
    for folder in infrastructure_image_folders:
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
                #for ann in annotations_coco:
                #    all_object_ids[folder].append(ann['object_id'])
            except:
                print(f"No annotations for image {img}")
                os.remove(f"{save_images_to}/{str(idx).zfill(6)}.png")
            annotations_mot = pd.concat([annotations_mot, convert_coco_to_mot(annotations_coco, idx)], ignore_index=True)
        track_id_sorted_mot_annotations = annotations_mot.sort_values(by=['track_id', 'frame_number'])
        track_id_sorted_mot_annotations['track_id'] = track_id_sorted_mot_annotations['track_id'] - track_id_sorted_mot_annotations['track_id'].min() + 1 
        #print(track_id_sorted_mot_annotations)

        save_gt_to = f"{data_output_path}/{folder}/gt"
        os.makedirs(save_gt_to, exist_ok=True)
        track_id_sorted_mot_annotations.to_csv(f"{save_gt_to}/gt.txt", sep=',', header=False, index=False)

        save_det_to = f"{data_output_path}/{folder}/det"
        os.makedirs(save_det_to, exist_ok=True)
        no_tracks_sorted_mot_annotations = copy.deepcopy(track_id_sorted_mot_annotations)
        no_tracks_sorted_mot_annotations['track_id'] = -1
        no_tracks_sorted_mot_annotations.to_csv(f"{save_det_to}/det.txt", sep=',', header=False, index=False)

    #for k,v in all_object_ids.items():
    #    all_object_ids[k] = set(v)
    #print(all_object_ids)
    