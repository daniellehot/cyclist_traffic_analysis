import cv2
import os, shutil
import argparse
import pandas as pd
import json
from pycocotools.coco import COCO
import random
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser("Visualizer")
    parser.add_argument("--tracks", type=str, default=None)
    parser.add_argument("--gt_tracks", type=str, default=None)
    #parser.add_argument("--draw_mot_gt", action="store_true")
    parser.add_argument("--seq", type=str, default=None)

    parser.add_argument("--detections", type=str, default=None)
    parser.add_argument("--gt_json", type=str, default=None)
    parser.add_argument("--img_root", type=str, default=None)
    
    parser.add_argument("-o", "--output", type=str, default=None)
    return parser.parse_args()


class Visualizer():
    # drawing settings
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    #text_thickness = 2

    @classmethod #TODO implement cat_color_map functionality
    def draw_mot_sequence(cls, tracks_txt, seq, color=None, output=None):
        # "frame_id", "track_id", "bb_left", "bb_top", "bb_width", "bb_height"
        tracks = pd.read_csv(tracks_txt, header=None)
        if color is None:
            color = cls.green
        if output is not None:
            if os.path.exists(output):
                shutil.rmtree(output)
            os.makedirs(output)

        annotated_mot_seq = []
        for i in range(1, len(os.listdir(seq))+1):
            image_path = cls.__get_image_extension(os.path.join(seq, str(i).zfill(6)))
            print(f"Processing image {image_path}")
            img = cv2.imread(image_path)
            frame_tracks = tracks[tracks[0] == i].to_numpy()
            
            track_ids = frame_tracks[:, 1]
            left = frame_tracks[:, 2]
            top = frame_tracks[:, 3]
            right = frame_tracks[:, 2]+frame_tracks[:, 4]
            bottom = frame_tracks[:, 3]+frame_tracks[:, 5]

            for y1, x1, y2, x2, id in zip(left, top, right, bottom, track_ids):
                bbox = list(map(int, [y1, x1, y2, x2]))
                img = cls.draw_bounding_box(bbox, img, color)
                text = f"{int(id)}"
                img = cls.draw_text(bbox, text, img, color)
            annotated_mot_seq.append(img)
            
            if output is not None:
                output_path = os.path.join(output, os.path.basename(image_path))
                cv2.imwrite(output_path, img)
                print(f"Saving image {output_path}")
        return annotated_mot_seq


    @classmethod #TODO implement cat_color_map functionality
    def draw_json_detections(cls, detections_json, gt_json, img_root, color = None, output = None):
        with open(detections_json, 'r') as file:
            detections = pd.DataFrame(json.load(file))
        gt = COCO(gt_json)
        categories = pd.DataFrame(gt.loadCats(gt.getCatIds()))
        image_ids = detections['image_id'].unique()

        if color is None:
            color = cls.red
        if output is not None:
            if os.path.exists(output):
                shutil.rmtree(output)
            os.makedirs(output)

        images_with_detections = []

        for id in image_ids:
            image_detections = detections[detections["image_id"] == id]
            image_path = os.path.join(img_root, gt.loadImgs([id])[0]['file_name'])
            print(f"Processing image {image_path}")
            img = cv2.imread(image_path)
            bboxes = image_detections['bbox'].to_list()
            category_ids = image_detections['category_id'].to_list()
            scores = image_detections['score'].to_list()
            for bbox, category_id, score in zip(bboxes, category_ids, scores):
                y1 = bbox[0]
                x1 = bbox[1]
                y2 = bbox[0]+bbox[2]
                x2 = bbox[1]+bbox[3]
                bbox = list(map(int, [y1, x1, y2, x2]))
                img = cls.draw_bounding_box(bbox, img, color)
                category = categories.loc[categories['id'] == category_id, 'name'][0]
                text = f"{str(category)}"
                img = cls.draw_text(bbox, text, img, color)    
            images_with_detections.append(img)

            if output is not None:
                output_path = os.path.join(output, os.path.basename(image_path))
                cv2.imwrite(output_path, img)
                print(f"Saving image {output_path}")
        return images_with_detections
    

    @classmethod
    def draw_coco_annotations(cls, gt_json, output=None, samples=10):
        coco = COCO(gt_json)
        images = coco.loadImgs(coco.getImgIds())
        categories = coco.loadCats(coco.getCatIds())
        cat_color_map = cls.generate_category_color_map(categories_df=pd.DataFrame(categories))
        images = random.choices(images, k=samples)
        test_or_train = "train" if "train" in gt_json else "test"
        for image in images:
            image_id = image['id']
            image_annotations = pd.DataFrame(coco.loadAnns(coco.getAnnIds(imgIds=image_id)))
            data_dir = gt_json.replace(f"annotations/{test_or_train}.json", "")
            file_name = os.path.join(data_dir, test_or_train, image['file_name'])
            print(f"Path {file_name} Exists {os.path.exists(file_name)}")
            print(f"Processing image {file_name}")
            img = cv2.imread(file_name)

            bboxes = image_annotations['bbox'].to_list()
            category_ids = image_annotations['category_id'].to_list()
            for bbox, category_id in zip(bboxes, category_ids):
                y1 = bbox[0]
                x1 = bbox[1]
                y2 = bbox[0]+bbox[2]
                x2 = bbox[1]+bbox[3]
                bbox = list(map(int, [y1, x1, y2, x2]))
                img = cls.draw_bounding_box(bbox, img, cat_color_map[category_id])
            cv2.imshow("draw_coco_annotations", img)
            cv2.waitKey()


    @classmethod #TODO
    def draw_model_output(cls, predictions, img, categories=None):
        for pred in predictions:
            x1, y1, x2, y2, obj_conf, class_conf, class_pred = pred
            bbox = list(map(int, [x1, y1, x2, y2]))
            cls.draw_bounding_box(bbox, img, cls.red)
        return img


    @staticmethod
    def __get_image_extension(path):
        image_extensions = [".png", ".jpg", ".jpeg"]
        for extension in image_extensions:
            filename = path + extension
            if os.path.exists(filename):
                return filename


    @staticmethod #TODO finish functionality for object_id, class_id, track_id, confidence 
    def draw_bounding_box(bbox, img, color, thickness = 2, object_id=None, class_id=None, track_id=None, confidence=None):
        left_top = (bbox[0], bbox[1])
        right_bottom = (bbox[2], bbox[3])
        cv2.rectangle(img, left_top, right_bottom, color, thickness)
        return img
    

    @staticmethod
    def draw_text(bbox, text, img, color, font = cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness = 1):
        # Calculate text size
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size[0], text_size[1]
        # Calculate the position to put text (in the middle of the bounding box)
        text_x = bbox[0] + (bbox[2] - bbox[0]) // 2 - text_width // 2
        text_y = bbox[1] + (bbox[3] - bbox[1]) // 2 + text_height // 2
        # Put text on the image
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness)
        return img
        

    @staticmethod
    def generate_category_color_map(categories_df=None, legend=True):
        df = categories_df
        ids = df.id.to_list()
        generate_random_color = lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cat_color_map = {cat_id:generate_random_color() for cat_id in ids}
        
        # Generate category legend (ChatGPT)
        id_name_map = dict(zip(df.id, df.name)) 
        if legend:
            circle_radius = 10
            margin = 10
            text_height = 20  # Adjust text height based on your font and size
            legend_width = 2 * circle_radius + 2 * margin + max([cv2.getTextSize(str(id_name_map[cat_id]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for cat_id in ids])
            legend_height = len(ids) * (2 * circle_radius + margin) + margin

            legend_img = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
            legend_img.fill(255)  # Fill with white background
            
            y = margin
            for cat_id, color in cat_color_map.items():
                cv2.circle(legend_img, (circle_radius + margin, y + circle_radius), circle_radius, color, -1)
                cv2.putText(legend_img, str(id_name_map[cat_id]), (2 * circle_radius + 2 * margin, y + text_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                y += 2 * circle_radius + margin
            
            cv2.imshow("Legend", legend_img)
        return cat_color_map 
        

    @staticmethod
    def generate_id_category_map(categories_df=None):
        df = categories_df
        id_name_map = dict(zip(df.id, df.name))
        return id_name_map
    

if __name__=="__main__":
    #print(make_parser())
    args = make_parser()
    #predicted_tracks = Visualizer.draw_mot_sequence(tracks_txt=args.tracks, seq=args.seq, output=args.output)
    #gt_tracks = Visualizer.draw_mot_sequence(tracks_txt=args.gt_tracks, seq=args.seq)
    #Visualizer.draw_json_detections(detections_json=args.detections, gt_json=args.gt_json, img_root=args.img_root, output=args.output)
    Visualizer.draw_coco_annotations(args.gt_json)