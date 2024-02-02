import os
#from pycocotools.coco import COCO
#import numpy as np
#import copy
import random
from detectron2.data import MetadataCatalog, DatasetCatalog, datasets
from detectron2.structures import BoxMode
#import shutil
#from tabulate import tabulate


class TrafficDataset:
    def __init__(self, dataset_root_folder):
        self.root_dir = dataset_root_folder
        self.train_split = f"{dataset_root_folder}/train"
        self.val_split = f"{dataset_root_folder}/valid"
        self.test_split = f"{dataset_root_folder}/test"
        self.splits = ["traffic_dataset_train", "traffic_dataset_validation", "traffic_dataset_test"]
        self.split_folder_dict = {
            'traffic_dataset_train': self.train_split, 
            'traffic_dataset_validation': self.val_split, 
            'traffic_dataset_test': self.train_split
        }
        
        self.objects = ['bicycle', 'bus', 'car', 'motorbike', 'person']
        self.image_id_map = self._generate_image_id_map(self.train_split, self.val_split, self.test_split)

    
    def _generate_image_id_map(self, train_split, val_split, test_split):
        id = 1
        image_id_map = {}

        for split in [train_split, val_split, test_split]:
            images = os.listdir(f"{split}/images")
            for image in images:
                image_id_map[f"{split}/images/{image}"] = id
                id += 1
        return image_id_map
                

    def read_yolo_annotations(self, file_path):
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
    
    
    def _get_bounding_box(self, annotation, image_height, image_width):
        x_center = int(annotation['x_center'] * image_width)
        y_center = int(annotation['y_center'] * image_height)
        width = int(annotation['width'] * image_width)
        height = int(annotation['height'] * image_height)

        # Calculate bounding box coordinates
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        return x_min, y_min, x_max, y_max


    def register_split(self, path):
        images =  os.listdir(f"{path}/images")
        annotation_files = [img.replace(".jpg", ".txt") for img in images]
        
        dataset_dicts = []
        for image, annotation_file in zip(images, annotation_files):
            annotations = self.read_yolo_annotations(f"{path}/labels/{annotation_file}")
            
            record = {}
            image_path = f"{path}/images/{image}"
            record["file_name"] = image_path
            record['image_id'] = self.image_id_map[image_path]

            image = cv2.imread(image_path)
            record['height'] = image.shape[0]
            record['width'] = image.shape[1] 

            detectron2_annotations = []
            for annotation in annotations:
                detectron2_annotations.append({
                    'segmentation': [],
                    'bbox': self._get_bounding_box(annotation, record['height'], record['width']), 
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': annotation['class_label']
                    })
            record["annotations"] = detectron2_annotations
            dataset_dicts.append(record)
        return dataset_dicts
    

    def register_dataset(self):
        for split in self.splits:
            DatasetCatalog.register(split, lambda folder_path=self.split_folder_dict[split]: self.register_split(folder_path))
            MetadataCatalog.get(split).set(thing_classes=self.objects)
            MetadataCatalog.get(split).set(evaluator_type="coco")
            

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Define the argument for the YAML file path
    parser.add_argument('-f', '--dataset_root_folder', type=str, help='Path to the dataset folder.')
    parser.add_argument('-s', '--save', action='store_true', help='Indicate if you would like to save the dataset as COCO JSON files.')
    parser.add_argument('-o', '--output', type=str, help='Where to save the COCO JSON files.')
    args = parser.parse_args()
    return args


def draw_samples(dataset_name, window_name="dataset", number_of_samples = 10):
    print("draw_samples")
    dataset = DatasetCatalog.get(dataset_name)
    print("DatasetCatalog")
    metadata = MetadataCatalog.get(dataset_name)
    print("MetadataCatalog")
    for d in random.sample(dataset, number_of_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(window_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


if __name__=="__main__":
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import yaml
    import argparse

    args = parse_arguments()

    dataset = TrafficDataset(args.dataset_root_folder)
    dataset.register_dataset()
    
    for d in DatasetCatalog.list():
        print(d)

    draw_samples("traffic_dataset_train")