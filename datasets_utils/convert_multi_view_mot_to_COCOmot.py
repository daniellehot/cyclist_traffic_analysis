import os, shutil
import numpy as np
import json
from PIL import Image
import argparse
from convert_multi_view_COCO_to_mot import SPLITS_DEFINITION

# TODO test.json can be filled with the annotations unlike the MOT datasets

# Better mix logic, e.g., randomly 70/20/10 pick from a combined set of all images 
# This would require rewriting some of the downstream logic
#SPLITS_DEFINITION = {
#    'train': ["0", "1000", "2000"],
#    'val': ["3000"],
#    'test': ["2659"]
#}


def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="MOT folder")
    parser.add_argument("-c", "--categories_txt", type=str, help="Path to the categories.txt file")
    return parser.parse_args()


def generate_category_id_list(categories_txt):
    with open(categories_txt, 'r') as f:
        categories = f.read().splitlines()

    # Create a dictionary with category names as keys and unique identifiers as values
    categories_ids_list = []
    for idx, category_name in enumerate(categories):
        categories_ids_list.append({
            'id': idx,
            'name': category_name
        })
    return categories_ids_list


def main(args):
    data_root_path = args.input
    output_path = os.path.join(data_root_path, 'annotations')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    if args.categories_txt is not None:
        categories_ids_list = generate_category_id_list(args.categories_txt)
    else:
        categories_ids_list = [{'id': 1, 'name': 'vehicle'}]


    for split in SPLITS_DEFINITION.keys():
        if split != "test":
            data_path = f"{data_root_path}/train"
        else:
            data_path = f"{data_root_path}/test"
        
        print(f"Preparing {split} split")
        output_path_JSON = os.path.join(output_path, f"{split}.json")
        out = {'images': [], 
               'annotations': [], 
               'videos': [],
               'categories': categories_ids_list
            }
        
        seqs = SPLITS_DEFINITION[split]
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1

        for seq in sorted(seqs):
            print(f"    Processing sequence {seq}")
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            images = os.listdir(img_path)
            num_images = len([image for image in images if '.png' in image])  # half and half
            image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                img = Image.open(os.path.join(data_path, '{}/img1/{:06d}.png'.format(seq, i + 1)))
                width, height = img.size
                image_info = {'file_name': '{}/img1/{:06d}.png'.format(seq, i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])
                if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                    continue
                track_id = int(anns[i][1])
                if not track_id == tid_last:
                    tid_curr += 1
                    tid_last = track_id
                ann_cnt += 1
                category_id = 1
                ann = {'id': ann_cnt,
                       'category_id': category_id,
                       'image_id': image_cnt + frame_id,
                       'track_id': tid_curr,
                       'bbox': anns[i][2:6].tolist(),
                       'conf': float(anns[i][6]),
                       'iscrowd': 0,
                       'area': float(anns[i][4] * anns[i][5])}
                out['annotations'].append(ann)

            image_cnt += num_images
        json.dump(out, open(output_path_JSON, 'w'))        
        

if __name__=='__main__':
    main(parse_args())