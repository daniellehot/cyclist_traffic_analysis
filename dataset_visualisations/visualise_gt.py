import cv2 
import os
import argparse
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images in a folder.')
    # Define the command-line argument
    parser.add_argument('-f', '--folder_path', type=str, help='Path to the folder containing images.')
    # Parse the command-line arguments
    return parser.parse_args()


def read_yolo_annotations(file_path):
    annotations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        data = line.strip().split()

        if len(data) == 5:  # YOLO format has 5 values for each object (class, x_center, y_center, width, height)
            class_label = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])

            annotations.append({
                'class_label': class_label,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

    return annotations


def draw_annotations(image, annotations, objects, color_map):
    image_height, image_width, _ = image.shape
    for annotation in annotations:
        class_label = annotation['class_label']
        x_center = int(annotation['x_center'] * image_width)
        y_center = int(annotation['y_center'] * image_height)
        width = int(annotation['width'] * image_width)
        height = int(annotation['height'] * image_height)

        # Calculate bounding box coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw bounding box on the image
        color = color_map[class_label] 
        thickness = 2
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Display class label
        label = objects[class_label]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        image = cv2.putText(image, label, (x_min, y_min - 5), font, font_scale, color, font_thickness)
    return image


def main(args):
    images =  os.listdir(f"{args.folder_path}/images")
    annotation_files = [img.replace(".jpg", ".txt") for img in images]
    
    traffic_objects = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    color_map = [(255, 255, 0), 
                 (255, 0, 255), 
                 (0, 255, 255),
                 (0, 0, 255),
                 (0, 255, 0),
                 ]
    
    object_dictionary = {}
    for idx, traffic_object in enumerate(traffic_objects):
        object_dictionary[idx] = traffic_object

    annotated_images = []
    # YOLO format <object-class> <x> <y> <width> <height>
    for img, ann in zip(images, annotation_files):
        print(f"annotating image {args.folder_path}/images/{img}")
        image = cv2.imread(f"{args.folder_path}/images/{img}")
        annotations = read_yolo_annotations(f"{args.folder_path}/labels/{ann}")
        annotated_images.append(draw_annotations(image, annotations, traffic_objects, color_map))
    
    for img in random.sample(annotated_images, 10):
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__=="__main__":
    main(parse_arguments())