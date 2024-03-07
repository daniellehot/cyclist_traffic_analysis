import argparse
import os, shutil
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import copy

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.tracker.byte_tracker import BYTETracker

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX Inference on Single Image")
    parser.add_argument("-f", "--exp_file", type=str, help="Path to experiment file")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model")
    parser.add_argument("-c", "--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("-i", "--image_path", type=str, help="Path to input image")
    parser.add_argument("--save_dir", type=str, help="Directory to save output images")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser.parse_args()


def load_model(exp, ckpt, rank=0):
    if ckpt is None:
        ckpt_file = os.path.join(exp.output_dir, exp.exp_name, "best_ckpt.pth.tar")
        print(f"Loading checkpoint {ckpt_file}")
    else:
        ckpt_file = ckpt
    model = exp.get_model()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()
    loc = "cuda:{}".format(rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    return model


def draw_detections(img, detections):
    image = copy.deepcopy(img)

    if detections[0] is not None:
        predictions =  detections[0].cpu().numpy()
        for pred in predictions:
            x1, y1, x2, y2, obj_conf, clqass_conf, class_pred = pred
            top_left = (int(y1), int(x1))
            bottom_right = (int(y2), int(x2))
            # Draw bounding box rectangle
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image


#def draw_tracks(img, detections, ids, scores):
def draw_tracks(img, targets):
    image = copy.deepcopy(img)

    # Iterate through detections, IDs, and scores
    for target in targets:
        bbox = target.tlbr
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        track_id = target.track_id
        # Draw bounding box
        # Write ID and score near the bounding box
        text_position = (int(bbox[0]/2 + bbox[2]/2), int(bbox[1]/2 + bbox[3]/2))
        cv2.putText(image, f"{track_id}", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image


def read_images(path):
    if os.path.isfile(path):
        return [path]
    else:
        images = [os.path.join(path, file) for file in os.listdir(path)]
        return sorted(images)


def inference_image(model, exp, img):
    transform = transforms.Compose([
        transforms.Resize((exp.input_size[0], exp.input_size[1])),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).cuda().unsqueeze(0) 
    with torch.no_grad():
        output = model(img_tensor)
        output = postprocess(output, 1)
        #output = postprocess(output, 1)[0].cpu().numpy()
    #img_with_boxes = draw_boxes_on_image(img, output)
    #save_path = os.path.join(save_dir, os.path.basename(image_path))
    #img_with_boxes.save(save_path)
    #print(f"Saved image with detection results: {save_path}")
    return output


def track(tracker, detections, args):
    # run tracking
    if detections[0] is not None:
        online_targets = tracker.update(detections[0], (640, 1024), (640, 1024))
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        #return online_targets, online_tlwhs, online_ids, online_scores
        return online_targets


def main(args):
    exp = get_exp(args.exp_file, args.model_name)
    
    if args.save_dir is not None:
        #save_dir = args.save_dir
        save_dir = os.path.join(exp.output_dir, exp.exp_name, args.save_dir)
    else:
        save_dir = os.path.join(exp.output_dir, exp.exp_name, "inference")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)    
    os.makedirs(save_dir, exist_ok=True)
    
    counter = 0
    model = load_model(exp, args.ckpt_path)
    tracker = BYTETracker(args)
    img_list = read_images(args.image_path)
    
    for idx, image_path in enumerate(img_list):
        if image_path.endswith(IMAGE_EXTENSIONS):
            img = Image.open(image_path).convert('RGB')
            img_np = cv2.imread(image_path)
        else:
            continue

        print(f"Processing image {image_path} {idx+1}/{len(img_list)}")
        detections = inference_image(model, exp, img)
        #img_detections = draw_detections(img_np, detections)
        
        #targets, tlwhs, ids, scores = track(tracker, detections, args)
        targets = track(tracker, detections, args)
        img_tracks = draw_tracks(img_np, targets)

        #output_img = np.hstack((img_detections, 
        #                        np.zeros((img_np.shape[0], 5, 3), dtype=np.uint8), 
        #                        img_tracks
        #                        ))
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, img_tracks)
        print(f"Saved image with detection results: {save_path}")

if __name__ == "__main__":    
    main(parse_args())
