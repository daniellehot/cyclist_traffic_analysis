import torch
#from torch.cuda.amp import autocast
import os, shutil
import numpy as np
import cv2
from PIL import Image
import argparse
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# detectron2
#from detectron2.config import get_cfg
#from detectron2.projects.deeplab import add_deeplab_config
#from detectron2.data import MetadataCatalog
#from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import create_ddp_model
from detectron2.config import instantiate

# shared
from train_utils import get_config


class Predictor:
    def __init__(self, configuration, weights):
        self.model = instantiate(configuration.model)
        self.model.to(configuration.train.device)
        self.model = create_ddp_model(self.model)
        DetectionCheckpointer(self.model).load(weights)
        self.model.eval()

    def __call__(self, original_image, augmentations=None):  
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            height, width = original_image.shape[:2]
            
            if augmentations != None:
                image = augmentations.get_transform(original_image).apply_image(original_image)
            else:
                image = original_image
            
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
        

def resize_image(image, scale):
        height, width, _ = image.shape
        target_dim = (int(width*scale), int(height*scale))
        return cv2.resize(image, target_dim, interpolation=cv2.INTER_AREA)


def concatenate_images(img1, img2, bar_height=10):
    black_bar = np.zeros((bar_height, img1.shape[1], img1.shape[2]), dtype=img1.dtype)
    return np.vstack((img1, black_bar, img2))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', required=True, type=str, help='Path to a YAML file')
    parser.add_argument('--input', required=True, type=str, help='Path to a folder with images') 
    parser.add_argument('--output', required=True, type=str, help='Output folder')
    args = parser.parse_args()
    return args


def main(args):
    print(args)

    with open(args.yaml, "r") as f:
        experiment_configuration = yaml.safe_load(f)

    # Create output directory 
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    # Config
    configuration = get_config(experiment_configuration, add_augmentations=False)

    # Predictor
    predictor = Predictor(configuration, os.path.join(experiment_configuration['output_dir'], 'model_final.pth'))
    #predictor = Predictor(configuration, configuration.train.init_checkpoint)
    img = np.array(Image.open(args.input))
    print(img.shape)
    print(predictor(img))

    """
    model_config =  yaml.safe_load(open(args.yaml))
    
    dataset = Autofish.instance_from_yaml(model_config)
    dataset.register_all_autofish_splits()
    config, classes = get_config(model_config=model_config)
    
    predictor = Predictor(cfg=config)
    
    os.makedirs(os.path.join(".", args.output), exist_ok=True)
    
    color_map = dict(zip(classes, VISUALIZER_COLORS))
    generate_legend(color_map, args.output)
    # Convert color_map to BGR as we are using opencv as our image library
    for k,v in color_map.items():
        v_reversed = tuple(list(v)[::-1])
        color_map[k] = v_reversed

    images = os.listdir(args.input)

    # Set confidence level. 
    if args.confidence != -1:
        confidence = args.confidence
    elif model_config.get('confidence'):
        confidence = model_config['confidence']
    else:
        print("No confidence level was provided")
        exit()
    
    # Draw labels
    if args.no_labels:
        draw_text = False
    else:
        draw_text= True

    for idx, image in enumerate(images):
        image_filename = os.path.join(args.input, image)
        print(f"working on {image_filename}")
        image = cv2.imread(image_filename)

        #resize image to the training size
        #can also be done with the predictor using resize augmentations as an input
        image = resize_image(image, model_config['test_augmentations']['scale'])
        
        predictions = predictor(image, conversion_function= lambda img: img[:, :, ::-1])
        pred_instances = predictions['instances'].to('cpu')
        pred_instances = pred_instances[pred_instances.scores > confidence]
        pred_masks = pred_instances.pred_masks.numpy()
        pred_labels = pred_instances.pred_classes.numpy()
        pred_labels = [classes[id] for id in pred_labels]
        pred_scores = pred_instances.scores.numpy()

        img_with_predictions = Visualizer.draw(image=image.copy(), 
                                               masks=pred_masks, 
                                               scores=pred_scores, 
                                               labels=pred_labels, 
                                               draw_text=draw_text,
                                               color_map=color_map,
                                               text_scale=model_config['test_augmentations']['scale']
                                               )
        
        img_to_save = concatenate_images(image, img_with_predictions)
        
        #output_image = image_filename.split("/")[-1]
        #output_image = output_image.replace(".tiff", ".png")
        #output_filename = os.path.join(".", args.output, output_image)
    
        output_filename = os.path.join(".", args.output, f"img_{idx}.png")
        cv2.imwrite(output_filename, img_to_save)
    """

if __name__=="__main__":
    main(parse_arguments())
