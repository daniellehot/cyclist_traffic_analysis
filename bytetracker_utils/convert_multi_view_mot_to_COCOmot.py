import os, shutil
import numpy as np
import json
from PIL import Image
import argparse

# TODO test.json can be filled with the annotations unlike the MOT datasets

# Better mix logic, e.g., randomly 70/20/10 pick from a combined set of all images 
# This would require rewriting some of the downstream logic
SPLITS_DEFINITION = {
    'train': ["0", "1000", "2000"],
    'val': ["3000"],
    'test': ["2659"]
}


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
    data_path = f"{data_root_path}/data"
    output_path = os.path.join(data_root_path, 'annotations')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    categories_ids_list = generate_category_id_list(args.categories_txt)

    for split in SPLITS_DEFINITION.keys():
        print(f"Preparing {split} split")
        output_path_JSON = os.path.join(output_path, f"{split}.json")

        out = {'images': [], 
               'annotations': [], 
               'videos': [],
               'categories': categories_ids_list
            }
        
        #category_dict = {
        #    'id': -1, 
        #    'name': 'none'
        #}

        seqs = SPLITS_DEFINITION[split]
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1

        for seq in sorted(seqs):
            #if '.DS_Store' in seq:
            #    continue
            #if 'mot' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
            #    continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            images = os.listdir(img_path)
            num_images = len([image for image in images if '.png' in image])  # half and half

            #if HALF_VIDEO and ('half' in split):
            #    image_range = [0, num_images // 2] if 'train' in split else \
            #                  [num_images // 2 + 1, num_images - 1]
            #else:
            #    image_range = [0, num_images - 1]
            image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                #img = cv2.imread(os.path.join(data_path, '{}/img1/{:06d}.png'.format(seq, i + 1)))
                #height, width = img.shape[:2]
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
            print('Processing sequence {}: {} images'.format(seq, num_images))
            
            if split != 'test':
                ann_path = os.path.join(seq_path, 'gt/gt.txt')
                det_path = os.path.join(seq_path, 'det/det.txt')
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                
                """
                if CREATE_SPLITTED_ANN and ('half' in split):
                    anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if int(anns[i][0]) - 1 >= image_range[0] and
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32) 
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
                    fout = open(gt_out, 'w')
                    for o in anns_out:
                        fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                    int(o[6]), int(o[7]), o[8]))
                    fout.close()
                if CREATE_SPLITTED_DET and ('half' in split):
                    dets_out = np.array([dets[i] for i in range(dets.shape[0])
                                         if int(dets[i][0]) - 1 >= image_range[0] and
                                         int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, 'det/det_{}.txt'.format(split))
                    dout = open(det_out, 'w')
                    for o in dets_out:
                        dout.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), float(o[2]), float(o[3]), float(o[4]), float(o[5]),
                                    float(o[6])))
                    dout.close()
                """

                #print('{} ann images'.format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    #cat_id = int(anns[i][7])
                    ann_cnt += 1

                    """
                    if not ('15' in DATA_PATH):
                        #if not (float(anns[i][8]) >= 0.25):  # visibility.
                            #continue
                        if not (int(anns[i][6]) == 1):  # whether ignore.
                            continue
                        if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                            continue
                        if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored person
                            category_id = -1
                        else:
                            category_id = 1  # pedestrian(non-static)
                            if not track_id == tid_last:
                                tid_curr += 1
                                tid_last = track_id
                    else:
                        category_id = 1
                    """

                    if not track_id == tid_last:
                        tid_curr += 1
                        tid_last = track_id

                    ann = {'id': ann_cnt,
                           'category_id': int(anns[i][7]),
                           'image_id': image_cnt + frame_id,
                           'track_id': tid_curr,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                    out['annotations'].append(ann)

            image_cnt += num_images
            #print(tid_curr, tid_last)
            print(f"Current track id {tid_curr} and the last track id {tid_last}")
        print('loaded {} split with {} images and {} annotation \n'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(output_path_JSON, 'w'))        
        

if __name__=='__main__':
    main(parse_args())