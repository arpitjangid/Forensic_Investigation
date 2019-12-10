import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data_path = "../data/FID-300"
divided_data_path = os.path.join(data_path, "divided")

ref_path_orig = os.path.join(data_path, "references")
ref_path = os.path.join(divided_data_path, "references")

train_path_orig = os.path.join(data_path, "tracks_cropped_train")
val_path_orig = os.path.join(data_path, "tracks_cropped_val")

train_path = os.path.join(divided_data_path, "tracks_cropped_train")
val_path = os.path.join(divided_data_path, "tracks_cropped_val")

if not os.path.exists(divided_data_path):
    os.makedirs(divided_data_path)

if not os.path.exists(train_path):
    os.makedirs(train_path)
    
if not os.path.exists(val_path):
    os.makedirs(val_path)
    
if not os.path.exists(ref_path):
    os.makedirs(ref_path)

flist = os.listdir(ref_path_orig)
flist.sort()

for f in flist:
    id = int(f[:5])
    src = os.path.join(ref_path_orig, f)
    I = skimage.io.imread(src)
    mid = int(I.shape[0]/2)
    I_U = I[:mid, :]
    I_B = I[mid:, :]

    dst_U = os.path.join(ref_path, "{:05d}.png".format(2*id-1))
    dst_B = os.path.join(ref_path, "{:05d}.png".format(2*id))
    skimage.io.imsave(dst_U, I_U)
    skimage.io.imsave(dst_B, I_B)

flist = os.listdir(train_path_orig)
flist.sort()

label_file = open(data_path + "label_table_train.csv")
label_map = np.loadtxt(label_file,delimiter=',',dtype='int')
label_dict = dict((a,b) for a,b in label_map)
label_map = np.zeros((0,2)).astype('int')

for f in flist:
    src = os.path.join(train_path_orig, f)
    I = skimage.io.imread(src)
    id = int(f[:5])
    
    if I.shape[0]/I.shape[1] > 2:
        mid = int(I.shape[0]/2)

        I_U = I[:mid, :]
        dst_U = os.path.join(train_path, "{:05d}.png".format(2*id-1))
        skimage.io.imsave(dst_U, I_U)

        I_B = I[mid:, :]
        dst_B = os.path.join(train_path, "{:05d}.png".format(2*id))
        skimage.io.imsave(dst_B, I_B)
        label_map = np.vstack((label_map, [2*id-1, 2*label_dict[id]-1]))
        label_map = np.vstack((label_map, [2*id, 2*label_dict[id]]))
        
    else:
        dst_U = os.path.join(train_path, "{:05d}.png".format(2*id-1))
        skimage.io.imsave(dst_U, I_U)
        label_map = np.vstack((label_map, [2*id-1, 2*label_dict[id]-1]))

label_path = os.path.join(divided_data_path, "label_table_train.csv")
np.savetxt(label_path, label_map, fmt="%d", delimiter=",")

flist = os.listdir(val_path_orig)
flist.sort()

label_file = open(data_path + "label_table_val.csv")
label_map = np.loadtxt(label_file,delimiter=',',dtype='int')
label_dict = dict((a,b) for a,b in label_map)
label_map = np.zeros((0,2)).astype('int')

for f in flist:
    src = os.path.join(val_path_orig, f)
    I = skimage.io.imread(src)
    id = int(f[:5])
    
    if I.shape[0]/I.shape[1] > 2:
        mid = int(I.shape[0]/2)

        I_U = I[:mid, :]
        dst_U = os.path.join(val_path, "{:05d}.png".format(2*id-1))
        skimage.io.imsave(dst_U, I_U)

        I_B = I[mid:, :]
        dst_B = os.path.join(val_path, "{:05d}.png".format(2*id))
        skimage.io.imsave(dst_B, I_B)
        label_map = np.vstack((label_map, [2*id-1, 2*label_dict[id]-1]))
        label_map = np.vstack((label_map, [2*id, 2*label_dict[id]]))
        
    else:
        dst_U = os.path.join(val_path, "{:05d}.png".format(2*id-1))
        skimage.io.imsave(dst_U, I_U)
        label_map = np.vstack((label_map, [2*id-1, 2*label_dict[id]-1]))

label_path = os.path.join(divided_data_path, "label_table_val.csv")
np.savetxt(label_path, label_map, fmt="%d", delimiter=",")