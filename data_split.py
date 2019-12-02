import os
import csv
import numpy as np
import random
import shutil 

data_path = "../data/FID-300/"
probe_folder = os.path.join(data_path,"tracks_cropped")
probe_train = os.path.join(data_path,"tracks_cropped_train")
probe_val = os.path.join(data_path,"tracks_cropped_val")

if(not os.path.exists(probe_train)):
    os.makedirs(probe_train)

if(not os.path.exists(probe_val)):
    os.makedirs(probe_val)

flist = os.listdir(probe_folder)
flist.sort()

fraction_train = 0.8
num_files = len(flist)
num_train = int(fraction_train*num_files)

label_file = open(data_path+'label_table.csv')
label_map = np.loadtxt(label_file,delimiter=',',dtype='int')

random.seed(701)
train_ids = random.sample(list(np.arange(num_files)),num_train)
train_ids.sort()
val_ids = list(set(np.arange(num_files))-set(train_ids))
val_ids.sort()

label_map_train = label_map[train_ids]
label_map_val = label_map[val_ids]

for id in train_ids:
    src = os.path.join(probe_folder,"{:05d}".format(id+1)+".jpg")
    dst = os.path.join(probe_train,"{:05d}".format(id+1)+".jpg")
    shutil.copyfile(src, dst) 

for id in val_ids:
    src = os.path.join(probe_folder,"{:05d}".format(id+1)+".jpg")
    dst = os.path.join(probe_val,"{:05d}".format(id+1)+".jpg")
    shutil.copyfile(src, dst) 

# save csv files
label_train_path = os.path.join(data_path,"label_table_train.csv")
label_val_path = os.path.join(data_path,"label_table_val.csv")

np.savetxt(label_train_path,label_map_train,fmt="%d", delimiter=",")
np.savetxt(label_val_path,label_map_val,fmt="%d", delimiter=",")
