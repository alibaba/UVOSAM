from metaseg import sahi_sliced_predict, SahiAutoSegmentation
import numpy as np 
import torch 
from PIL import Image 
import cv2 
import os
import glob
import json 
import argparse

# get the template palette
template_img = 'assets/template.png' 
im = Image.open(template_img)
template_palette = im.getpalette()

def get_args_parser():
    parser = argparse.ArgumentParser('Set evaluation for davis', add_help=False)
    parser.add_argument('--davis_path', default='datasets/davis', type=str)
    parser.add_argument('--track_res', type=str)
    parser.add_argument('--save_path', default='test',type=str)
    return parser

def process_anns(img, masks, out_file):
    h, w, _ = img.shape
    num_mask = len(masks)
    real_idx = 0
    out_mask = np.zeros([h, w]).astype(np.uint8)
    for idx in range(num_mask):
        cur_mask = masks[idx][0]

        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        real_idx += 1
        out_mask[np.where(cur_mask == 1)] = real_idx

    im_save = out_mask.astype(np.uint8)
    im_save = Image.fromarray(im_save, mode='P')
    im_save.putpalette(template_palette)
    save_path = out_file + '.png'
    im_save.save(save_path)
    print("Saved: {}".format(save_path))

def main(davis_path, track_res, save_path):
    track_res = json.load(open(track_res,'r'))
    track_per_video = {}
    for item in track_res:
        if item['video_id'] not in track_per_video.keys():
            track_per_video[item['video_id']] = [item['boxes']]
        else:
            track_per_video[item['video_id']].append(item['boxes'])

    test_split = '{}/ImageSets/2017/val.txt'.format(davis_path)
    f = open(test_split,'r')
    split_lines = f.readlines()
    test_names = [line.strip() for line in split_lines]

    img_dir = "{}/JPEGImages/480p".format(davis_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for i, name_1 in enumerate(test_names):
        cur_img_dir = os.path.join(img_dir, name_1)
        if not os.path.isdir(os.path.join(save_path,name_1)):
            os.mkdir(os.path.join(save_path,name_1))
        cur_img_list = sorted(glob.glob(cur_img_dir + "/*.*g"))

        for img_idx, img_path in enumerate(cur_img_list):
            img_array = np.array(Image.open(img_path))
            boxes = []
            for item in track_per_video[name_1]:
                if item[img_idx] == None:
                    boxes.append([0,0,0,0])
                else:
                    boxes.append(item[img_idx])
            masks = SahiAutoSegmentation().predict(
                    source=img_path,
                    model_type="vit_h",
                    input_box=boxes,
                    multimask_output=False,
                    random_color=True,
                    show=False,
                    save=True,)
            name_2 = os.path.basename(img_path).split('.')[0]
            masks = [mask.cpu().numpy() for mask in masks]
            out_file = os.path.join(save_path, name_1, name_2)
            process_anns(img_array, masks, out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args.davis_path, args.track_res, args.save_path)