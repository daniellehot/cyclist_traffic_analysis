import cv2 
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images in a folder.')
    # Define the command-line argument
    parser.add_argument('-f', '--folder_path', type=str, help='Path to the folder containing images.')
    # Parse the command-line arguments
    return parser.parse_args()


def main(args):
    images = os.listdir(f"{args.folder_path}/images")
    annotations = [img.replace("jpg", "txt") for img in images]
    
    traffic_objects = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    object_dictionary = {}
    for idx, traffic_object in enumerate(traffic_objects):
        object_dictionary[idx] = traffic_object

    for img, ann in images, annotations:
        print("TODO")


    
if __name__=="__main__":
    main(parse_arguments())