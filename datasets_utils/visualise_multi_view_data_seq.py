from pycocotools.coco import COCO
import argparse
import os, shutil
import cv2
import random


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


def draw_coco_annotations(image_path, annotations):
    print(f"Drawing COCO annotations")
    # Open the image
    image = cv2.imread(image_path)
    
    # Iterate over annotations and draw bounding boxes
    for annotation in annotations:
        bbox = annotation['bbox']  # Bounding box in format [x_min, y_min, width, height]
        bbox = list(map(int, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]))
        object_id = annotation['object_id']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), generate_random_color(object_id), 2)
        text = f"{object_id}"
        draw_text(image, text, bbox)
    # Show the image with bounding boxes
    return image


def main(args):
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    
    if args.jpg:
        sequence = sorted([img for img in os.listdir(args.seq) if img.endswith(".jpg")])
    if args.png:
        sequence = sorted([img for img in os.listdir(args.seq) if img.endswith(".png")])

    coco = COCO(args.input)
    image_ids_map = generate_image_ids_map(coco, sequence)

    for idx, img in enumerate(sequence):
        try:
            annotation_ids = coco.getAnnIds(imgIds=image_ids_map[img])
            annotations = coco.loadAnns(annotation_ids)
            image = draw_coco_annotations(image_path = f"{args.seq}/{img}", annotations = annotations)
            cv2.imwrite(f"{args.output}/{idx}.png", image)
        except:
            print(f"No annotations for image {img}")

if __name__=="__main__":
    main(parse_args())