#from pycocotools.coco import COCO
import argparse
import os, shutil
#from PIL import Image, ImageDraw
import cv2
import random
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="MOT folder")
    #parser.add_argument("-c", "--coco", type=str, help="COCO annotations")
    parser.add_argument("-o", "--output", type=str, help="Output folder")
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


def generate_random_color(seed):
    # Set seed for reproducibility
    random.seed(seed)
    
    # Generate random RGB values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    # Return the random color as a tuple (R, G, B)
    return (r, g, b)


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


def draw_mot_annotations(image_path, annotations):
    print("Drawing mot annotations")
    image = cv2.imread(image_path)

    # Iterate over annotations and draw bounding boxes
    for _, annotation in annotations.iterrows():
        bbox = [annotation[i] for i in range(2, 6)]  # Bounding box in format [x_min, y_min, width, height]
        bbox = list(map(int, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]))
        track_id = annotation[1]
        category_id = annotation[7]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), generate_random_color(int(track_id)), 2)
        text = f"{track_id}"
        draw_text(image, text, bbox)
    return image


def main(args):
    print(args)
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    
    sequence = sorted([img for img in os.listdir(f"{args.input}/img1")])
    annotations = f"{args.input}/gt/gt.txt"  
    # panda_frame_data "frame_id", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "?", "??"]  
    df = pd.read_csv(annotations, header=None)

    for idx, img in enumerate(sequence):
        print(f"Processing image {img} {idx+1}/{len(sequence)}")
        if img.endswith(".jpg"):
            img_no = int(img.replace(".jpg", ""))

        if img.endswith(".png"):
            img_no = int(img.replace(".png", ""))
        
        annotations = df[df[0] == img_no]
        annotated_image = draw_mot_annotations(image_path=f"{args.input}/img1/{img}",
                                annotations=annotations
                                )
        cv2.imwrite(f"{args.output}/{idx}.png", annotated_image)


if __name__=="__main__":
    main(parse_args())