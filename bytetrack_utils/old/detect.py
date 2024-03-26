import argparse
import os, shutil
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX Inference on Single Image")
    parser.add_argument("-f", "--exp_file", type=str, help="Path to experiment file")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model")
    parser.add_argument("-c", "--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("-i", "--image_path", type=str, help="Path to input image")
    parser.add_argument("--save_dir", type=str, help="Directory to save output images")
    args = parser.parse_args()
    return args


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


def draw_boxes_on_image(image, predictions):
    draw = ImageDraw.Draw(image)
    # Process output to draw boxes on the image
    # Assuming the output is in the format of [class, x_center, y_center, width, height, confidence]
    font = ImageFont.load_default()
    #print(predictions)
    for pred in predictions:
        x1, y1, x2, y2, obj_conf, class_conf, class_pred = pred

        # Draw bounding box rectangle
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

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


def read_images(path):
    if os.path.isfile(path):
        return [path]
    else:
        images = [os.path.join(path, file) for file in os.listdir(path)]
        return sorted(images)


def inference_image(model, exp, image_path, save_dir):
    img_list = read_images(image_path)
    for idx, image_path in enumerate(img_list):
        if image_path.endswith(IMAGE_EXTENSIONS):
            img = Image.open(image_path).convert('RGB')
        else:
            continue
        print(f"Processing image {image_path} {idx+1}/{len(img_list)}")

        transform = transforms.Compose([
            transforms.Resize((exp.input_size[0], exp.input_size[1])),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).cuda().unsqueeze(0) 
        with torch.no_grad():
            output = model(img_tensor)
            output = postprocess(output, 1)[0].cpu().numpy()
        img_with_boxes = draw_boxes_on_image(img, output)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        img_with_boxes.save(save_path)
        print(f"Saved image with detection results: {save_path}")


def main(args):
    exp = get_exp(args.exp_file, args.model_name)
    model = load_model(exp, args.ckpt_path)
    
    if args.save_dir is not None:
        #save_dir = args.save_dir
        save_dir = os.path.join(exp.output_dir, exp.exp_name, args.save_dir)
    else:
        save_dir = os.path.join(exp.output_dir, exp.exp_name, "inference")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)    
    os.makedirs(save_dir, exist_ok=True)

    inference_image(model, exp, args.image_path, save_dir)

if __name__ == "__main__":     
    main(parse_args())
