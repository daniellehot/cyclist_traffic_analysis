import os
#from pycocotools.coco import COCO
import numpy as np
import copy
import random
import argparse
import yaml
#import cv2
from PIL import Image
#import shutil
from tabulate import tabulate

from detectron2.data import MetadataCatalog, DatasetCatalog, datasets
from detectron2.structures import BoxMode


class TrafficDataset:
    def __init__(self, dataset_root_folder, objects_ids_map=None):
        self.root_dir = dataset_root_folder
        self.train_split = f"{dataset_root_folder}/train"
        self.val_split = f"{dataset_root_folder}/valid"
        self.test_split = f"{dataset_root_folder}/test"
        self.splits = ["traffic_dataset_train", "traffic_dataset_validation", "traffic_dataset_test"]
        self.split_folder_dict = {
            'traffic_dataset_train': self.train_split, 
            'traffic_dataset_validation': self.val_split, 
            'traffic_dataset_test': self.test_split
        }
        
        if objects_ids_map is None:
            self.objects_ids_map = {'bicycle':0, 'bus':1, 'car':2, 'motorbike':3, 'person':4}
            self.objects = list(self.objects_ids_map.keys())
            self.object_ids = list(self.objects_ids_map.values())
        else:
            self.objects_ids_map = objects_ids_map
            self.objects = list(self.objects_ids_map.keys())
            self.object_ids = list(self.objects_ids_map.values())

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


    def _get_image_dimensions(self, file_path):
        with Image.open(file_path) as img:
            width, height = img.size
        return width, height


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

            record['width'], record['height'] = self._get_image_dimensions(image_path)                  
            #record['height'] = 640
            #record['width'] = 640 

            detectron2_annotations = []
            for annotation in annotations:
                detectron2_annotations.append({
                    'segmentation': [],
                    'bbox': self._get_bounding_box(annotation, record['height'], record['width']), 
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': self.object_ids[annotation['class_label']] #remap labels
                    #'category_id': annotation['class_label']
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
    parser.add_argument('-y', '--yaml', type=str, help="Path to a yaml file.")
    parser.add_argument('-d', '--dataset_root_folder', type=str, help="Path to the dataset folder.")
    parser.add_argument('--visualize-dataset', action='store_true', help="Visualize registered dataset.")
    parser.add_argument('--number-of-samples-to-visualize', type=int, default=10, help="How many images to save. Default is 10.")
    parser.add_argument('--save-JSON', action='store_true', help="Indicate if you would like to save the dataset as COCO JSON files.")
    parser.add_argument('--output-JSON', type=str, help="Where to save the COCO JSON files.")

    args = parser.parse_args()
    return args


def draw_samples(dataset_name, output_dir="./dataset_samples", number_of_samples=10):
    from detectron2.utils.visualizer import Visualizer
    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(random.sample(dataset, number_of_samples)):
        print(f"Drawing {dataset_name} Sample {idx+1}/{number_of_samples}")
        img = Image.open(sample["file_name"])
        visualizer = Visualizer(np.array(img), metadata=metadata, scale=1)
        out = visualizer.draw_dataset_dict(sample)
        output_path = os.path.join(output_dir, f"sample_{idx + 1}.png")
        out.save(output_path)


def print_dataset_statistics(dataset_instance):
    def _generate_hist_of_instances_per_class(_split, _class_names):
        dataset_dict = DatasetCatalog.get(_split)
        num_classes = len(_class_names)
        hist_bins = np.arange(num_classes + 1)
        histogram = np.zeros((num_classes,), dtype=int)
        for entry in dataset_dict:
            annos = entry["annotations"]
            classes = np.asarray(
                [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=int
            )
            if len(classes):
                assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
                assert (
                    classes.max() < num_classes
                ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
            histogram += np.histogram(classes, bins=hist_bins)[0]

        histogram = [v for _, v in enumerate(histogram)]
        return histogram
     
    histograms = []
    table_labels = copy.deepcopy(dataset_instance.objects)
    table_labels.append("total")
    for split in dataset_instance.splits:
        hist = _generate_hist_of_instances_per_class(split, dataset_instance.objects)
        hist.append(sum(hist))
        histograms.append(hist)
        histograms[-1].insert(0, split)
    
    #compute autofish_total (train+val+test)
    total_hist = [0 for i in range(len(table_labels))]
    total_hist.insert(0, "traffic_dataset_total")
    for hist in histograms:
        if hist[0] in ["traffic_dataset_train", "traffic_dataset_validation", "traffic_dataset_test"]:
            for i in range(1, len(hist)):
                total_hist[i] += hist[i]
    histograms.append(total_hist)

    table_labels.insert(0, "dataset_split")
    print(tabulate(histograms, headers=table_labels))


def main(args):
    print(args)
    with open(args.yaml, "r") as f:
        experiment_configuration = yaml.safe_load(f)

    dataset_root_folder = experiment_configuration['dataset_folder'] or args.dataset_root_folder
    traffic_dataset = TrafficDataset(dataset_root_folder)
    traffic_dataset.register_dataset()
    
    #for d in DatasetCatalog.list():
    #    print(d)

    print_dataset_statistics(traffic_dataset)

    if args.visualize_dataset:
        draw_samples("traffic_dataset_train", number_of_samples=args.number_of_samples_to_visualize)
        draw_samples("traffic_dataset_validation", number_of_samples=args.number_of_samples_to_visualize)
        draw_samples("traffic_dataset_test", number_of_samples=args.number_of_samples_to_visualize)


if __name__=="__main__":
    main(parse_arguments())