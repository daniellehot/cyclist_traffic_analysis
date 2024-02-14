from pycocotools.coco import COCO
import argparse
import os
from PIL import Image, ImageDraw
import random
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="MOT folder")
    parser.add_argument("-c", "--coco", type=str, help="COCO annotations")
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
        

def draw_bounding_boxes_mot(image_path, annotations, output):
    print(f"Drawing {image_path}")
    # Open the image
    image = Image.open(image_path)
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    
    # Iterate over annotations and draw bounding boxes
    for _, annotation in annotations.iterrows():
        bbox = [annotation[i] for i in range(2, 6)]  # Bounding box in format [x_min, y_min, width, height]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        object_id = annotation[1] # track id
        print(object_id)
        draw.rectangle(bbox, outline=generate_random_color(object_id), width=2)  # Draw bounding box with red outline
        text_position = (bbox[0] + bbox[2] + 5, bbox[1])
        draw.text(text_position, str(object_id), fill="black")
    # Show the image with bounding boxes
    image.save(output)


def draw_bounding_boxes_coco(image_path, annotations, output):
    print(f"Drawing {image_path}")
    # Open the image
    image = Image.open(image_path)
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    
    # Iterate over annotations and draw bounding boxes
    for annotation in annotations:
        bbox = annotation['bbox']  # Bounding box in format [x_min, y_min, width, height]
        object_id = annotation['object_id']
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        draw.rectangle(bbox, outline=generate_random_color(object_id), width=2)  # Draw bounding box with red outline
    
    # Show the image with bounding boxes
    image.save(output)


def main(args):
    print(args)
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    sequence = sorted([img for img in os.listdir(f"{args.input}/img1")])
    annotations = f"{args.input}/gt/gt.txt"  
    # panda_frame_data "frame_id", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "?", "??"]  
    df = pd.read_csv(annotations, header=None)

    for idx, img in enumerate(sequence):
        if img.endswith(".jpg"):
            img_no = int(img.replace(".jpg", ""))

        if img.endswith(".png"):
            img_no = int(img.replace(".png", ""))
        
        annotations = df[df[0] == img_no]
        draw_bounding_boxes_mot(image_path=f"{args.input}/img1/{img}",
                                annotations=annotations,
                                output = f"{args.output}/{idx}.png"
                                )


if __name__=="__main__":
    main(parse_args())