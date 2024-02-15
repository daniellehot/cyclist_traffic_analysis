from pycocotools.coco import COCO
import argparse
import cv2
import os, shutil
import pandas as pd
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="Folder with sequences")
    parser.add_argument("-c", "--coco", type=str, help="COCO annotations")
    parser.add_argument("-o", "--output", type=str, help="Output folder")
    return parser.parse_args()


class MultiViewDataset:
    def __init__(self, coco, mot, COCOmot):
        self.coco = 
        self.mot = 
        self.COCOmot = 

    

def generate_random_color(seed):
    # Set seed for reproducibility
    random.seed(int(seed))    
    # Generate random RGB values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    # Return the random color as a tuple (R, G, B)
    return (r, g, b)


def tag_image(image, tag):
    # Define the font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    # Get the size of the text
    text_size = cv2.getTextSize(tag, font, font_scale, font_thickness)[0]
    # Calculate the position to draw text (in the upper right corner)
    image_height, image_width, _ = image.shape
    text_x = image_width - text_size[0] - 10  # 10 pixels from the right edge
    text_y = text_size[1] + 1 # 10 pixels from the top edge
    # Draw the text on the image
    cv2.putText(image, tag, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)


def draw_text(image, text, bbox):
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
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)


def draw_coco_annotations(image_path, annotations):
    print(f"Drawing COCO annotations")
    # Open the image
    image = cv2.imread(image_path)
    tag_image(image, "COCOmot")
    
    # Iterate over annotations and draw bounding boxes
    for annotation in annotations:
        bbox = annotation['bbox']  # Bounding box in format [x_min, y_min, width, height]
        bbox = list(map(int, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]))
        track_id = annotation['track_id']
        category_id = annotation['category_id']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), generate_random_color(track_id), 2)
        text = f"{track_id}"
        draw_text(image, text, bbox)
    # Show the image with bounding boxes
    return image


def draw_mot_annotations(image_path, annotations):
    print("Drawing mot annotations")
    image = cv2.imread(image_path)
    tag_image(image, "mot")

    # Iterate over annotations and draw bounding boxes
    for _, annotation in annotations.iterrows():
        bbox = [annotation[i] for i in range(2, 6)]  # Bounding box in format [x_min, y_min, width, height]
        bbox = list(map(int, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]))
        track_id = annotation[1]
        category_id = annotation[7]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), generate_random_color(track_id), 2)
        text = f"{track_id}"
        draw_text(image, text, bbox)
    return image


def main(args):
    #annotations_coco = f"{args.coco}"
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    #data_root_folder = f"{args.input}/data/"
    data_root_folder = f"{args.input}"
    coco = COCO(args.coco)

    images = coco.loadImgs(coco.getImgIds())
    for image_info in images:
        image_path = os.path.join(data_root_folder, image_info["file_name"])
        annotation_ids = coco.getAnnIds(imgIds=image_info["id"])
        annotations_coco = coco.loadAnns(annotation_ids)
        
        seq_name = image_info["file_name"].split("/")[0]
        if not os.path.exists(os.path.join(args.output, seq_name)):
            os.makedirs(os.path.join(args.output, seq_name))

        if image_info["file_name"].endswith(".png"):
            frame = int(image_info["file_name"].split("/")[-1].replace(".png", ""))
        if image_info["file_name"].endswith(".jpg"):
            frame = int(image_info["file_name"].split("/")[-1].replace(".jpg", "")) 
        annotations_mot_path = os.path.join(data_root_folder, seq_name, "gt", "gt.txt")
        annotations_mot = pd.read_csv(annotations_mot_path, header=None)
        annotations_mot = annotations_mot[annotations_mot[0] == frame]

        print(f"Processing image {image_path}")
        image_coco = draw_coco_annotations(image_path, annotations_coco)
        image_mot = draw_mot_annotations(image_path, annotations_mot)

        # Stack images vertically
        divider = np.zeros((image_coco.shape[0], 10, image_coco.shape[2]))
        stacked_image = np.hstack((image_coco, divider, image_mot))
        cv2.imwrite(os.path.join(args.output, seq_name, f"{frame}.jpg"), stacked_image)


if __name__=="__main__":
    main(parse_args())