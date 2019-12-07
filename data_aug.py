import numpy as np 
import os
import shutil 
import skimage.io
import skimage.transform
# from PIL import Image

if __name__ == "__main__":
    path = "../data/FID-300/"
    imgs_path = path + "tracks_cropped_train/"

    label_file = open(path + "label_table_train.csv")
    label_map = np.loadtxt(label_file,delimiter=',',dtype='int')

    # num_aug = 5 # including original, 4 rotations
    aug_label_path = path + "label_table_train_aug.csv"
    aug_folder = path + "tracks_cropped_train_aug/"
    if(not os.path.exists(aug_folder)):
        os.makedirs(aug_folder)
    
    flist = os.listdir(imgs_path)
    flist.sort()

    rotations = [-20, -10, 10, 10]
    count = 301

    for i, fname in enumerate(flist):
        
        image_path = os.path.join(imgs_path, fname)
        I = skimage.io.imread(image_path)
        # I = Image.open(image_path)
        label = label_map[i,1]

        # copy original file
        dst = os.path.join(aug_folder, fname)
        shutil.copyfile(image_path, dst) 

        for angle in rotations:
            I_r = skimage.transform.rotate(I, angle)
            # file_path = image_path.split('.jpg')[0] + 'rot_{}.jpg'.format(angle)
            file_path = os.path.join(aug_folder, "{:05d}".format(count)+".jpg")
            print("count", count)
            label_map = np.vstack((label_map, [count, label]))
            count += 1
            skimage.io.imsave(file_path, I_r)
            
        
    np.savetxt(aug_label_path,label_map,fmt="%d", delimiter=",")





