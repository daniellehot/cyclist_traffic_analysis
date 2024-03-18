from pycocotools.coco import COCO
import json
import pandas as pd
import numpy as np
import argparse
import os, shutil

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

TO_SKIP = ["Egensevej", "Hjorringvej-3", "Hjorringvej-4", "Hobrovej", "Ostre-4"]
TEST_SEQUENCES = ["Drone"]

class DatasetCompiler:
    def __init__(self, json_files, output_dir=None):
        self.json_files = json_files

        if output_dir is None:
            self.output_dir = "traffic_dataset"
        else:
            self.output_dir = output_dir
        
        #if os.path.exists(self.output_dir):
        #    remove_output_dir = input(f"{self.output_dir} exists. Overwrite? y/n ")
        #    if remove_output_dir == "y" or remove_output_dir == "Y":
        #        shutil.rmtree(self.output_dir)
        #os.makedirs(self.output_dir)
            
        
        self.images_df = self.get_images_df()
        train_images, test_images = self.get_images(self.images_df)
        train_annotations, test_annotations = self.get_annotations()
        categories = self.get_categories()

        self.train_coco = {
            'images': train_images,
            'annotations' : train_annotations,
            'categories' : categories
        }

        self.test_coco = {
            'images': test_images,
            'annotations' : test_annotations,
            'categories' : categories
        }

        #TODO Save jsons
        #TODO Move and rename files 

    """
    @staticmethod
    def state(text, tab=0, tab_space="   "):
        empty_space = ""
        for i in range(tab):
            empty_space += tab_space
        state_str = f"{empty_space} {text}"
        print(state_str)    
    """
    
    def get_images_df(self):
        images = []
        for f in self.json_files:
            coco = COCO(f)
            images_coco = coco.loadImgs(coco.getImgIds())
            images.extend(images_coco)
        images_df = pd.DataFrame(images)
        images_df = self.remove_sequences_to_skip(images_df)
        images_df = self.add_new_image_ids(images_df)
        images_df = self.add_new_filenames(images_df)
        images_df = self.add_train_test_tag(images_df)
        return images_df

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
            new_filenames.append(os.path.join(seq_location+"-"+current_seq_dir, str(frame_id).zfill(6)+".png"))
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
    def get_images(images_df):
        images = images_df[images_df['test'] == False]
        images = images.drop(['id', 'license', 'file_name', 'test'], axis=1)
        images = images.rename(columns={"new_id":"id", "new_file_name":"file_name"})
        train_images = images
        images = images_df[images_df['test'] == True]
        images = images.drop(['id', 'license', 'file_name', 'test'], axis=1)
        images = images.rename(columns={"new_id":"id", "new_file_name":"file_name"})
        test_images = images
        return train_images, test_images
        
    def get_annotations(self):
        train_annotations = []
        test_annotations = []
        for f in self.json_files:
            coco = COCO(f)
            all_images = [img['file_name'] for img in coco.loadImgs(coco.getImgIds())]   
            for image in all_images:
                image_df = self.images_df[self.images_df['file_name'] == image] 
                print(image_df)
                image_id = image_df.iloc[:,2]
                image_new_id = image_df.iloc[:, 5]
                print(image_id, image_new_id)
                exit()
                
                #all_image_annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
                #for ann in all_image_annotations:
                #    ann['id'] = image_new_id
                
                #if image_df['test']:
                #    test_annotations.extend(all_image_annotations)
                #else:
                #    train_annotations.extend(all_image_annotations)
        exit()
        return train_annotations, test_annotations
        

    def get_categories(self):
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
        categories_df = categories_df.drop_duplicates(subset=['id', 'name'])
        categories_df = categories_df.sort_values(by='id')
        categories_df = categories_df[:-1]
        categories_df.loc[categories_df['name'] == 'lorry', 'name'] = 'truck'
        categories_df = categories_df.drop('other_names', axis=1)
        return categories_df.to_dict(orient='records')


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset complier")
    #parser.add_argument("-i", "--input", nargs="+", type=str, help="Annotation file")
    parser.add_argument("-i", "--input", nargs="+")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    dataset = DatasetCompiler(json_files=args.input)