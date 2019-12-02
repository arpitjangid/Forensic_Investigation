import numpy as np
import os
import skimage.io
import shutil 
import csv
# import cv2

root = "../data/FID-300/"
tracks_folder = root+"tracks_cropped/"
ref_folder = root + "references/"
label_file = open(root +'label_table.csv')
label_map = np.loadtxt(label_file,delimiter=',',dtype='int')

track_imgs = os.listdir(tracks_folder)
track_imgs.sort()

ref_imgs = os.listdir(ref_folder)
ref_imgs.sort()

out_folder = root + "tracks_cropped_squareshape/"
out_folder_oblong = root + "tracks_cropped_oblong/"

if(not os.path.exists(out_folder)):
    os.makedirs(out_folder)

if(not os.path.exists(out_folder_oblong)):
    os.makedirs(out_folder_oblong)

for i, fname in enumerate(track_imgs):
    image_path = os.path.join(tracks_folder, fname)
    I = skimage.io.imread(image_path)
    # I = cv2.imread(image_path)
    H,W = I.shape
    if(H/W < 2.0):
        # not oblong
        dst = os.path.join(out_folder,fname)
        shutil.copyfile(image_path, dst)
        
        gt_id = label_map[i,1]
        ref_path = os.path.join(ref_folder,"{:05d}".format(gt_id)+".png")
        ref_name = fname.split('.jpg')[0] + "_gt_" + "{:05d}".format(gt_id)+".png"
        dst_ref = os.path.join(out_folder,ref_name)
        shutil.copyfile(ref_path, dst_ref)
    else:
        out_name = fname.split('.jpg')[0] + "_ratio_" + "{:0.2f}".format(H/W)+".jpg"
        dst = os.path.join(out_folder_oblong,out_name)
        shutil.copyfile(image_path, dst)




