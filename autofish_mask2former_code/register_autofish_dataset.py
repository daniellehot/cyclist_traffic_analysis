import os
from pycocotools.coco import COCO
import numpy as np
import copy
import random
from detectron2.data import MetadataCatalog, DatasetCatalog, datasets
from detectron2.structures import BoxMode
import shutil
from tabulate import tabulate


#FILES_TO_EXCLUDE = ["00001.tiff", "00012.tiff", "00023.tiff", "00034.tiff", "00045.tiff", "00056.tiff", #jai rgb 
#                    "00001.png", "00012.png", "00023.png", "00034.png", "00045.png", "00056.png"] #rs rgb and depth

FILES_TO_EXCLUDE = ["00001", "00012", "00023", "00034", "00045", "00056"] #images with labels, no file format as these are true across different file formats

LABELS_CONFIGURATIONS = {
    'C1': {
        'classes': ["whiting", "cod", "haddock", "hake", "horse_mackerel", "other"],
        'id_species_map': {
            "whiting": 0,
            "cod": 1,
            "haddock": 2,
            "hake": 3,
            "horse_mackerel": 4,
            "saithe": 5,
            "other": 5,
        },
    },

    'C2': {
        'classes': ["whiting", "cod", "haddock", "other"],
        'id_species_map': {
            "whiting": 0,
            "cod": 1,
            "haddock": 2,
            "hake": 3,
            "horse_mackerel": 3,
            "saithe": 3,
            "other": 3,
        },
    },

    'C3': {
        'classes': ["cod-like", "other"],
        'id_species_map': {
            "whiting": 0,
            "cod": 0,
            "haddock": 0,
            "hake": 1,
            "horse_mackerel": 1,
            "saithe": 1,
            "other": 1,
        },
    },

    'C4': {
        'classes': ["fish"],
        'id_species_map': {
            "whiting": 0,
            "cod": 0,
            "haddock": 0,
            "hake": 0,
            "horse_mackerel": 0,
            "saithe": 0,
            "other": 0,
        },
    }
}


class Autofish:
    def __init__(self, root_dir, train_split, val_split, test_split, labels_configuration=None, classes=None, id_species_map=None):
        self.root_dir = root_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.test_mini_split = [10]
        self.exclusions = {}

         # if classes and id_species_map arguments are used explicitly, use these instead of the label_configuration argument  
        if classes != None and id_species_map != None:
            self.classes = classes
            self.id_species_map = id_species_map
        elif labels_configuration != None:
            self.classes = LABELS_CONFIGURATIONS[labels_configuration]['classes']
            self.id_species_map = LABELS_CONFIGURATIONS[labels_configuration]['id_species_map']  
        else:
            print("Provide label_configuration or classes and id_species_map argument as the input.")
            exit()
        
        # create splits, remove val split it no validation groups are specified
        self.jai_splits = ["autofish_train", "autofish_val", "autofish_test", "autofish_test_mini"]
        self.rs_splits = ["autofish_rs_train", "autofish_rs_val", "autofish_rs_test", "autofish_rs_test_mini"]
        self.splits = self.jai_splits + self.rs_splits
        if len(self.val_split) == 0:
            self.splits.remove("autofish_val")
            self.splits.remove("autofish_rs_val") 

        # create dictionary with information about each create split, e.g., camera type, image type, groups 
        self.split_configurations = self.build_split_configurations(self.splits)
       
        # create unique id for each image in our dataset
        self.filepath_id_map = {}
        self.build_filepath_id_map(camera="jai", image_type="rgb", file_format=".tiff")
        self.build_filepath_id_map(camera="rs", image_type="rgb", file_format=".png")
        
    
    @classmethod
    def instance_from_yaml(cls, yaml_config):
        #_yaml_config = copy.deepcopy(yaml_config)
        _yaml_config = yaml_config

        if _yaml_config.get('subsample_train_split') and _yaml_config['subsample_train_split']['subsample']:
            _yaml_config['train_split'] = random.sample(_yaml_config['train_split'], _yaml_config['subsample_train_split']['no_of_groups'])
            print(_yaml_config['train_split'])

        if "classes" and "id_species_map" in _yaml_config:
            instance = cls(
                root_dir=_yaml_config["root_dir"],
                train_split=_yaml_config["train_split"],
                val_split=_yaml_config["val_split"],
                test_split=_yaml_config["test_split"],
                classes=_yaml_config["classes"],
                id_species_map=_yaml_config["id_species_map"]
            )
        elif "labels_configuration" in _yaml_config:
            instance = cls(
                root_dir=_yaml_config["root_dir"],
                train_split=_yaml_config["train_split"],
                val_split=_yaml_config["val_split"],
                test_split=_yaml_config["test_split"],
                labels_configuration=_yaml_config["labels_configuration"]
            )
        
        if _yaml_config.get('exclusions'):
            instance.exclusions = _yaml_config['exclusions']

        return instance


    def build_split_configurations(self, splits):
        configurations = {}
        for split in splits:
            configurations[split] = {
                'camera': "",
                'image_type': "",
                'groups': 0
            }
            #camera
            if "rs" in split:
                configurations[split]['camera'] = "rs"
            else:
                configurations[split]['camera'] = "jai"
            #image type
            if "pc" in split:
                configurations[split]['image_type'] = "pc"
                exit()
            elif "depth" in split:
                configurations[split]['image_type'] = "depth"
                exit()
            else:
                configurations[split]['image_type'] = "rgb"
            #groups
            if split.endswith("train"):
                configurations[split]['groups'] = self.train_split
            elif split.endswith("val"):
                configurations[split]['groups'] = self.val_split
            elif split.endswith("test"):
                configurations[split]['groups'] = self.test_split
            elif split.endswith("test_mini"):
                configurations[split]['groups'] = self.test_mini_split
        
        return configurations


    def build_filepath_id_map(self, camera, image_type, file_format):
        groups = range(1, 26) #1 to 25
        images = range(1, 67) #1 to 66

        if len(self.filepath_id_map.values())==0:
            id = 1
        else:
            id = max(self.filepath_id_map.values())
        
        for group in groups:
            for image in images:
                image_name = str(image).zfill(5)
                if image_name not in FILES_TO_EXCLUDE:
                    filepath = os.path.join(self.root_dir, "group_"+str(group), camera, image_type, image_name+file_format)
                    if os.path.exists(filepath):            
                        self.filepath_id_map[filepath] = id
                        id += 1
        

    def get_category_id(self, cat_name):
        return self.id_species_map[cat_name.split("-")[1]]


    def build_category_annotations_map(self, anns):
        annotation_ids = []
        category_annotations_map = []
        for ann in anns:
            if ann["category_id"] not in annotation_ids:
                dict_obj = {
                    "category_id":-1,
                    "annotation_ids":[],
                    "segmentations":[]
                }
                annotation_ids.append(ann["category_id"])
                dict_obj["category_id"] = ann["category_id"]
                dict_obj["annotation_ids"].append(ann["id"])
                dict_obj["segmentations"].append(ann["segmentation"][0])
                category_annotations_map.append(dict_obj)
            else:
                idx = annotation_ids.index(ann["category_id"])
                category_annotations_map[idx]["annotation_ids"].append(ann["id"])
                category_annotations_map[idx]["segmentations"].append(ann["segmentation"][0])
        return category_annotations_map


    def compute_bbox(self, segmentations):
        polygons = []
        for polygon in segmentations:
            polygon = np.asarray(polygon)
            polygons.append(np.reshape(polygon, (-1, 2)))

        all_x_coords = []
        all_y_coords = []
        for polygon in polygons:
            x_coords, y_coords = zip(*polygon)
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)

        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)

        return x_min, y_min, x_max, y_max


    def register_split(self, camera, image_type, groups):
        camera_options = ["jai", "rs"] 
        image_types = ["rgb", "pc", "depth"]
        assert camera in camera_options and image_type in image_types, f"Camera or image_type unavailable. Allowed camera options are {camera_options}. Allowed image types are {image_types}" 
        
        gt_filepath = {
            'jai': {
                'rgb': 'annotations/coco/manual/dataset.json',
            },
            'rs' : {
                'rgb': 'annotations/coco/mapped_from_jai/dataset.json',
                'pc': "",
                'depth': "",
            }
        }

        image_size =  {
            'jai': {
                'height': 2056,
                'width': 2464,
            },
            'rs': {
                'height': 1080,
                'width': 1920,
            }
        }

        dataset_dicts = []
        for group in groups:
            #json_file = os.path.join(self.root_dir, "group_"+str(group), "jai", "annotations", "coco", "manual", "dataset.json")
            json_file = os.path.join(self.root_dir, "group_"+str(group), camera, gt_filepath[camera][image_type])
            coco_annotation = COCO(annotation_file=json_file) 
            images = coco_annotation.loadImgs(coco_annotation.getImgIds())
            
            if self.exclusions.get(group):
                exclusions = [f"{str(img).zfill(5)}" for img in self.exclusions.get(group)]
            else:
                exclusions = []

            for image in images:
                if image["file_name"].split(".")[0] in list(set(FILES_TO_EXCLUDE) | set(exclusions)):
                    #print("VAL_SPLIT Skipping image {} of group {}".format(image["file_name"], group))
                    pass
                else:
                    record = {}
                    #record["file_name"] = os.path.join(self.root_dir, "group_"+str(group), "jai", "rgb", image["file_name"]) 
                    record["file_name"] = os.path.join(self.root_dir, "group_"+str(group), camera, image_type, image["file_name"]) 
                    # Rewrite the image_id value otherwise ids are not unique
                    record["image_id"] = self.filepath_id_map[record["file_name"]]
                    # Set correct image dimensions because reading it from a dataset json will throw SizeMismatchError 
                    #TODO dataset.json for RealSense images has incorrect image dimensions
                    record["height"] = image_size[camera]["height"]
                    record["width"] = image_size[camera]["width"] 

                    # Get all annotations for the current image
                    ann_ids = coco_annotation.getAnnIds(imgIds=image["id"], iscrowd=None)
                    anns = coco_annotation.loadAnns(ann_ids)

                    category_annotations_map = self.build_category_annotations_map(anns)
                    annotation_objs = []
                    for map in category_annotations_map:
                        ann_cat_id = coco_annotation.loadCats([map["category_id"]])[0] #Zero index because the function always returns a list, even if there is only a single query id
                        annotation_objs.append({
                                "segmentation": map["segmentations"],
                                "bbox": self.compute_bbox(map["segmentations"]), 
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": self.get_category_id(ann_cat_id["name"])
                        })
                    record["annotations"] = annotation_objs
                    dataset_dicts.append(record)
        return dataset_dicts


    def register_autofish_split(self, split):
        assert split in self.splits, f"{split} is not available. Available splits are {self.splits}"
        camera = self.split_configurations[split]['camera']
        image_type = self.split_configurations[split]['image_type']
        groups = self.split_configurations[split]['groups']

        DatasetCatalog.register(split, lambda camera=camera, image_type=image_type, groups=groups: self.register_split(camera, image_type, groups))
        MetadataCatalog.get(split).set(thing_classes=self.classes)
        MetadataCatalog.get(split).set(evaluator_type="coco")
        dataset = DatasetCatalog.get(split)
        metadata = MetadataCatalog.get(split)
        return dataset, metadata
    

    def register_all_autofish_splits(self):
        for split in self.splits:
            self.register_autofish_split(split)
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Define the argument for the YAML file path
    parser.add_argument('-y', '--yaml', type=str, help='Path to the YAML file.')
    parser.add_argument('-s', '--save', action='store_true', help='Indicate if you are saving the dataset as a coco json file')
    parser.add_argument('-o', '--output', type=str, help='Where to save a coco json file')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize samples from the registered dataset')
    args = parser.parse_args()
    return args


def print_dataset_statistics(dataset_instance):
    histograms = []
    table_labels = copy.deepcopy(dataset_instance.classes)
    table_labels.append("total")
    for split in dataset_instance.splits:
        hist = generate_hist_of_instances_per_class(split, dataset_instance.classes)
        hist.append(sum(hist))
        histograms.append(hist)
        histograms[-1].insert(0, split)
    
    #compute autofish_total (train+val+test)
    total_hist = [0 for i in range(len(table_labels))]
    total_hist.insert(0, "autofish_total")
    for hist in histograms:
        if hist[0] in ["autofish_train", "autofish_val", "autofish_test"]:
            for i in range(1, len(hist)):
                total_hist[i] += hist[i]
    histograms.append(total_hist)

    table_labels.insert(0, "dataset_split")
    print(tabulate(histograms, headers=table_labels))


def generate_hist_of_instances_per_class(dataset, class_names):
    dataset_dict = DatasetCatalog.get(dataset)
    num_classes = len(class_names)
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
    

def draw_samples(dataset, metadata, window_name, number_of_samples = 3):
    for d in random.sample(dataset, number_of_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(window_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


if __name__=="__main__":
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import yaml
    import argparse

    args = parse_arguments()
    config = yaml.safe_load(open((args.yaml)))
    
    dataset = Autofish.instance_from_yaml(config)
    dataset.register_all_autofish_splits()

    # sanity check
    #for d in DatasetCatalog.list():
    #    print(d)

    print_dataset_statistics(dataset)

    if args.save:
        os.makedirs(args.output)
        for split in dataset.splits:
            datasets.convert_to_coco_json(split,output_file=os.path.join(args.output, f"{split}.json"), allow_cached=False)
        shutil.copy2(args.yaml, args.output)

    # TODO Fix visualisations
    if args.visualize:
        draw_samples(data, metadata, "jai_test_mini")
        draw_samples(rs_data, rs_metadata, "rs_test_mini")
        