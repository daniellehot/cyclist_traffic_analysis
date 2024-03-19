from pycocotools.coco import COCO
import json
import pandas as pd
import numpy as np
import argparse
import os, shutil
from tqdm import tqdm

"""
There are no object IDs for RainSnow dataset.
multi-view-data -> use all sequences and both perspectives
aau_rain_snow -> skip over super difficult images with bad weather
Skip - Egensevej, Hjorringvej-3, Hjorringvej-4, Hobrovej, Ostre-4

Purpose of this code is to create a new DETECTION dataset from a combination of multi-view-data and aau_RainSnow. 
Categories should be - car, motorcycle, truck, bicycle, person

Expected folder structure 
- annotations
    - test.json ()
    - train.json
- test (must have det.txt and gt.txt for each test sequence)
    - test_seq_0
        - det/det.txt 
        - gt/gt.txt 
        - img1/image_data (zerofilled to 6 digits, png)
    - test_seq_1
    - ... 
- train (training does not require tracking gt)
    - train_seq_0
        - det (empty)
        - gt (empty)
        - img1(zerofilled to 6 digits, png)
    - train_seq_1
    - ...

COCO dataset general format
images - list of dictionaries, dictionary keys are ['file_name', 'height', 'id', 'width'] where the 'id' refers to an image id
annotations - list of dictionaries, dict keys are ['area', 'bbox', 'category_id', 'id', 'image_id', 'iscrowd', 'segmentation']
    - 'id' refers to an annotation id 
    - 'segmentation' can be an empty list 
    - 'iscrowd' can be left empty
categories - list of dictionaries, dict keys are ['id', 'name', 'supercategory'] where 'id' refers to a category integer id
Custom keys can be added, such as 'object_id' in the multi-view-dataset

"""

#TO_SKIP = ["Egensevej", "Hjorringvej-3", "Hjorringvej-4", "Hobrovej", "Ostre-4"]
TO_SKIP = []
TEST_SEQUENCES = ["Drone"]

class DatasetCompiler:
    def __init__(self, json_files, output_dir=None):
        self.json_files = json_files

        if output_dir is None:
            self.output_dir = "traffic_dataset"
        else:
            self.output_dir = output_dir
                
        self.images_df, self.train_images, self.test_images = self.get_images()
        self.train_annotations, self.test_annotations = self.get_annotations()
        self.categories = self.get_categories()

        self.train_coco = {
            'images': self.train_images,
            'annotations' : self.train_annotations,
            'categories' : self.categories
        }

        self.test_coco = {
            'images': self.test_images,
            'annotations' : self.test_annotations,
            'categories' : self.categories
        }

        #TODO Save
        self.train_json, self.test_json = self.save_dataset()
        ##test whether jsons are valid COCO annotations
        #self.validate_coco_json(self.train_json)
        #self.validate_coco_json(self.test_json)
        ##TODO Move and rename files 
        
    def get_images(self):
        print("\nget_images")
        images = []
        root_dirs = []
        for f in self.json_files:
            coco = COCO(f)
            images_coco = coco.loadImgs(coco.getImgIds())
            images.extend(images_coco)
            root_dirs.extend(self.get_root_dirs(images_coco, f))
        images_df = pd.DataFrame(images)
        images_df['root_dir'] = root_dirs
        images_df = self.remove_sequences_to_skip(images_df)
        images_df = self.add_new_image_ids(images_df)
        images_df = self.add_new_filenames(images_df)
        images_df = self.add_train_test_tag(images_df)
        train_images, test_images = self.get_images_dict(images_df)
        print(f"Number of train images {len(train_images)}")
        print(f"Number of test images {len(test_images)}")
        return images_df, train_images, test_images

    @staticmethod
    def get_root_dirs(images_coco, json):
        # remove ***.json from a path
        json = json.replace(os.path.basename(json), "")
        # split path into list such that each path member can be compared
        json = json.split("/")
        dirs = []
        for img_data in images_coco:
            root_dir = ""
            img = img_data['file_name']
            for string in json:
                if string not in img:
                    root_dir = os.path.join(root_dir, string)
            dirs.append(root_dir)
        return dirs

    @staticmethod
    def remove_sequences_to_skip(df):
        files = df['file_name'].to_list()
        files_to_remove = []
        for idx, f in enumerate(files):
            seq_location = f.split("/")[-3]
            seq_dir = f.split("/")[-2]
            if seq_location in TO_SKIP or seq_dir in TO_SKIP:
                files_to_remove.append(idx)
        df = df.drop(files_to_remove)
        return df
    
    @staticmethod
    def add_new_image_ids(df):
        number_of_images = df.shape[0]
        new_ids = np.arange(number_of_images)
        df['new_id'] = new_ids
        return df
    
    @staticmethod
    def add_new_filenames(df):
        new_filenames = []
        files = df['file_name'].to_list()
        previous_seq_dir = None
        for f in files:
            seq_location = f.split("/")[-3]
            current_seq_dir = f.split("/")[-2]
            if current_seq_dir != previous_seq_dir:
                frame_id = 1
                previous_seq_dir = current_seq_dir
            else:
                frame_id += 1  
            new_filenames.append(os.path.join(seq_location+"-"+current_seq_dir, "img1", str(frame_id).zfill(6)+".png"))
        df['new_file_name'] = new_filenames
        return df
    
    @staticmethod
    def add_train_test_tag(df):
        files = df['file_name'].to_list()
        test = []
        for f in files:
            for seq_name in TEST_SEQUENCES:
                #print(f, seq_name, seq_name in f)
                if seq_name in f:
                    test.append(True)
                else:
                    test.append(False)
        df['test'] = test
        return df
    
    @staticmethod
    def get_images_dict(images_df):
        images = images_df[images_df['test'] == False]
        images = images.drop(['id', 'license', 'file_name', 'test', 'root_dir'], axis=1)
        images = images.rename(columns={"new_id":"id", "new_file_name":"file_name"})
        train_images = images.to_dict(orient='records')
        images = images_df[images_df['test'] == True]
        images = images.drop(['id', 'license', 'file_name', 'test', 'root_dir'], axis=1)
        images = images.rename(columns={"new_id":"id", "new_file_name":"file_name"})
        test_images = images.to_dict(orient='records')
        return train_images, test_images
        
    def get_annotations(self):
        print("\nget_annotations")
        train_annotations = []
        test_annotations = []
        annotation_counter = 1
        for f in self.json_files:
            coco = COCO(f)
            all_images = [img['file_name'] for img in coco.loadImgs(coco.getImgIds())]   
            for image in all_images:
                image_df = self.images_df[self.images_df['file_name'] == image]
                if image_df.empty:
                    continue
                image_id = int(image_df['id'])
                image_new_id = int(image_df['new_id'])
                all_image_annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
                for ann in all_image_annotations:
                    ann['image_id'] = image_new_id
                    ann['id'] = annotation_counter
                    annotation_counter += 1
                
                if image_df['test'].bool():
                    test_annotations.extend(all_image_annotations)
                else:
                    train_annotations.extend(all_image_annotations)
        print(f"Number of train annotations: {len(train_annotations)}")
        print(f"Number of test annotations: {len(test_annotations)}")
        return train_annotations, test_annotations
        
    def get_categories(self):
        print("\nget_categories")
        # Create a new category list, but only with categories that were annotated 
        categories = []
        for f in self.json_files:
            coco = COCO(f)
            annotations_coco = coco.loadAnns(coco.getAnnIds())
            annotations_df = pd.DataFrame(annotations_coco)
            categories_ids = annotations_df['category_id'].unique()
            categories_coco = coco.loadCats(coco.getCatIds(catIds=categories_ids))
            categories.extend(categories_coco)
        categories_df = pd.DataFrame(categories)
        categories_df = categories_df.sort_values(by='id')
        categories_df.loc[categories_df['name'] == 'lorry', 'name'] = 'truck'   
        categories_df = categories_df.drop_duplicates(subset=['id', 'name'])
        if 'other_names' in categories_df.columns:
            categories_df = categories_df.drop('other_names', axis=1)
        print(f"Categories:\n{categories_df}")
        return categories_df.to_dict(orient='records')

    def save_dataset(self):
        if os.path.exists(self.output_dir):
            remove_output_dir = input(f"\n{self.output_dir} exists. Overwrite? y/n ")
            if remove_output_dir == "y" or remove_output_dir == "Y":
                shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        annotations_dir = os.path.join(self.output_dir, "annotations")
        os.makedirs(annotations_dir)
        train_data_dir = os.path.join(self.output_dir, "train")
        os.makedirs(train_data_dir)
        test_data_dir = os.path.join(self.output_dir, "test")
        os.makedirs(test_data_dir)
        # Create json files
        train_json = os.path.join(annotations_dir, "train.json")
        data = self.train_coco
        with open(train_json, "w") as json_file:
            json.dump(data, json_file)
        
        test_json = os.path.join(annotations_dir, "test.json")
        data = self.test_coco
        with open(test_json, "w") as json_file:
            json.dump(data, json_file)
        
        # Copy images
        self.copy_files(df=self.images_df[self.images_df['test'] == False], copy_to=train_data_dir)
        self.copy_files(df=self.images_df[self.images_df['test'] == True], copy_to=test_data_dir)

        return train_json, test_json
    
    @staticmethod
    def copy_files(df, copy_to):
        files = df['file_name'].to_list()
        root_dirs = df['root_dir'].to_list()
        new_files = df['new_file_name'].to_list()

        with tqdm(total=len(files), desc="Copying images") as pbar:
            for f, r, nf in zip(files, root_dirs, new_files):
                copy_from = os.path.join(r, f)
                _copy_to = os.path.join(copy_to, nf)

                _copy_to_dir = _copy_to.replace(os.path.basename(_copy_to), "")
                os.makedirs(_copy_to_dir, exist_ok=True)
                os.makedirs(_copy_to_dir.replace("img1", "det"), exist_ok=True)
                os.makedirs(_copy_to_dir.replace("img1", "gt"), exist_ok=True)
                
                shutil.copy(copy_from, _copy_to)
                #if os.path.exists(copy_from):
                    #print(f"FROM {copy_from} TO {os.path.join(copy_to, nf)}")
                pbar.update(1)



def parse_args():
    parser = argparse.ArgumentParser(description="Dataset complier")
    #parser.add_argument("-i", "--input", nargs="+", type=str, help="Annotation file")
    parser.add_argument("-i", "--input", nargs="+")
    parser.add_argument("-o", "--output", type=str)
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    dataset = DatasetCompiler(json_files=args.input, output_dir=args.output)