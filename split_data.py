import os
import sys
import shutil
import glob
import os.path as osp

dev_num = 2500

def split_data(root, target_dir):
    img_files = glob.glob(root+"/*.jpg")
    dev_files = img_files[-dev_num:]
    print('dev_files:', dev_files)

    for dev_file in dev_files:
        shutil.move(dev_file, osp.join(target_dir, osp.basename(dev_file)))
        


if __name__ == "__main__":
    root = sys.argv[1]
    target_dir = sys.argv[2]
    split_data(root, target_dir)
