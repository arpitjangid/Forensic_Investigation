import os
import csv
import numpy as np
import random
import shutil 

def copy_files(src_dir, dst_dir, ids):
    for id in ids:
        src = os.path.join(src_dir,"{:05d}".format(id+1)+".jpg")
        dst = os.path.join(dst_dir,"{:05d}".format(id+1)+".jpg")
        shutil.copyfile(src, dst) 

if __name__ == "__main__":

    data_path = "../data/FID-300/"
    probe_folder = os.path.join(data_path,"tracks_cropped")
    probe_train = os.path.join(data_path,"tracks_cropped_train")
    probe_val = os.path.join(data_path,"tracks_cropped_val")
    probe_test = os.path.join(data_path,"tracks_cropped_test")

    if(not os.path.exists(probe_train)):
        os.makedirs(probe_train)

    if(not os.path.exists(probe_val)):
        os.makedirs(probe_val)

    if(not os.path.exists(probe_test)):
        os.makedirs(probe_test)

    flist = os.listdir(probe_folder)
    flist.sort()

    fraction_train = 0.7
    fraction_val = 0.1
    fraction_test = 0.2
    num_files = len(flist)
    num_train = int(fraction_train*num_files)
    num_val = int(fraction_val*num_files)

    label_file = open(data_path+'label_table.csv')
    label_map = np.loadtxt(label_file,delimiter=',',dtype='int')

    random.seed(10701)
    train_val_ids = random.sample(list(np.arange(num_files)),num_train+num_val)
    train_ids = train_val_ids[:num_train]
    train_ids.sort()
    val_ids = train_val_ids[num_train:]
    val_ids.sort()
    test_ids = list(set(np.arange(num_files))-set(train_val_ids))
    test_ids.sort()

    label_map_train = label_map[train_ids]
    label_map_val = label_map[val_ids]
    label_map_test = label_map[test_ids]

    copy_files(probe_folder, probe_train, train_ids)
    copy_files(probe_folder, probe_val, val_ids)
    copy_files(probe_folder, probe_test, test_ids)

    # save csv files
    label_train_path = os.path.join(data_path,"label_table_train.csv")
    label_val_path = os.path.join(data_path,"label_table_val.csv")
    label_test_path = os.path.join(data_path,"label_table_test.csv")

    np.savetxt(label_train_path,label_map_train,fmt="%d", delimiter=",")
    np.savetxt(label_val_path,label_map_val,fmt="%d", delimiter=",")
    np.savetxt(label_test_path,label_map_test,fmt="%d", delimiter=",")
