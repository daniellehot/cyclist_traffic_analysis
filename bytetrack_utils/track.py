import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

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


def draw_detections(image, detections):
    if detections[0] is not None:
        predictions =  detections[0].cpu().numpy()
        for pred in predictions:
            x1, y1, x2, y2, obj_conf, class_conf, class_pred = pred

            # Draw bounding box rectangle
            cv2.rectangle(image, (int(y1), int(x1)), (int(y2), int(x2)), (0, 255, 0), 2)

            # Construct label
            #label = f"{class_names[int(class_pred)]}: {class_conf:.2f}"
            #label = str(class_conf)

            # Calculate text size and position
            #text_width, text_height = draw.textsize(label, font=font)
            #text_x = x1
            #text_y = y1 - text_height - 5

            # Draw label text
            #draw.text((text_x, text_y), label, fill="green", font=font)
    return image


def draw_tracks(image, detections, ids, scores):
    cv2.rectangle(image, (20, 100), (40, 200), (0, 255, 0), 2)
    cv2.imshow("test", image)
    
    # Iterate through detections, IDs, and scores
    for bbox, id, score in zip(detections, ids, scores):
        # Extract bounding box coordinates
        #bbox = detection[:4]
        #print(detection)

        # Convert bounding box to integers 
        bbox_int = [int(coord) for coord in bbox]
        top, left, width, height = bbox_int
        bottom = top + height
        right = left + width
        # Draw bounding box
        cv2.rectangle(image, (top, left), (bottom, right), (0, 255, 0), 2)

        # Convert float score to string
        score_str = '{:.2f}'.format(score)

        # Write ID and score near the bounding box
        #cv2.putText(image, f'ID: {id}, Score: {score_str}', (bbox_int[0], bbox_int[1] - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def read_images(path):
    if os.path.isfile(path):
        return [path]
    else:
        images = [os.path.join(path, file) for file in os.listdir(path)]
        return sorted(images)


def inference_image(model, exp, img):
    transform = transforms.Compose([
        transforms.Resize((exp.input_size[1], exp.input_size[0])),
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
        return online_targets, online_tlwhs, online_ids, online_scores


def main(args):
    exp = get_exp(args.exp_file, args.model_name)
    
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(exp.output_dir, exp.exp_name, "tracking")
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
        targets, tlwhs, ids, scores = track(tracker, detections, args)
        img_detections = draw_detections(img_np, detections)
        img_tracks = draw_tracks(img_np, tlwhs, ids, scores)
        cv2.imshow("test", img_detections)
        cv2.waitKey()
        exit()

if __name__ == "__main__":    
    main(parse_args())
