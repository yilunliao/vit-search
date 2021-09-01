'''
    Generate sub-train and sub-val datasets for NAS.
'''


import os
import pickle
from pathlib import Path
from shutil import copyfile
import random


# Path to the original training set of ImageNet
_SOURCE_DIR     = '/data/datasets/imagenet-fast/data/imagenet/train'

# Path to the newly generated sub-train and sub-val sets used for NAS
_SUB_TRAIN_DIR  = '/code/imagenet/sub-train'
_SUB_VAL_DIR    = '/code/imagenet/sub-val'

_BUILD_PICKLE_PATH = False
_PICKLE_PATH = './sub_val_files.pickle'

_NUM_HOLDOUT_IMAGES = 25


if __name__ == '__main__':
    
    if _BUILD_PICKLE_PATH:
        random.seed(0)
        
        # extract the folders
        filenames = os.listdir(_SOURCE_DIR)
        folders = []
        for filename in filenames:  # loop through all the files and folders
            if os.path.isdir(os.path.join(os.path.abspath(_SOURCE_DIR), filename)):  # check whether the current object is a folder or not
                folders.append(filename)
    
        # extract the files
        holdout_files = {}
        index_f = 0
        for folder in folders:
            holdout_files[folder] = []
            filenames = os.listdir(os.path.join(os.path.abspath(_SOURCE_DIR), folder))
            filenames.sort()
            random.shuffle(filenames)
            
            for filename in filenames:
                if filename.endswith('.JPEG'):
                    holdout_files[folder].append(filename)
                if len(holdout_files[folder]) == _NUM_HOLDOUT_IMAGES:
                    break
                
            print('[{}]/[{}]'.format(index_f, len(folders)))
            index_f = index_f + 1
    
        # save the pickle file
        with open(_PICKLE_PATH, 'wb') as file_handle:
            pickle.dump(holdout_files, file_handle)
            
    else:
        # load the pickle file
        with open(_PICKLE_PATH, 'rb') as file_handle:
            holdout_files = pickle.load(file_handle)
    
        # check
        if len(holdout_files) != 1000:
            raise ValueError('There are not exactly 1000 folders (classes).')
        for folder in holdout_files:
            if len(holdout_files[folder]) != _NUM_HOLDOUT_IMAGES:
                raise ValueError(
                    'There are not exactly {} hold-out images for the folder {}.'.format(_NUM_HOLDOUT_IMAGES, folder))
    
        # create the holdout folder
        Path(_SUB_TRAIN_DIR).mkdir(parents=True, exist_ok=True)
        Path(_SUB_VAL_DIR).mkdir(parents=True, exist_ok=True)
    
        # move the files
        class_folders = []
        index_class = 0
        
        for dir_name in os.listdir(_SOURCE_DIR):
            if os.path.isdir(os.path.join(_SOURCE_DIR, dir_name)):
                class_folders.append(dir_name)
        
        for class_name in class_folders:
            assert class_name in holdout_files.keys()
            
            Path(os.path.join(os.path.abspath(_SUB_TRAIN_DIR), class_name)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(os.path.abspath(_SUB_VAL_DIR), class_name)).mkdir(parents=True, exist_ok=True)
            
            img_name_list = os.listdir(os.path.join(_SOURCE_DIR, class_name))
            for img_name in img_name_list:
                source_path = os.path.join(_SOURCE_DIR, class_name, img_name)
                dst_set = None
                if img_name in holdout_files[class_name]:
                    dst_set = _SUB_VAL_DIR
                else:
                    dst_set = _SUB_TRAIN_DIR
                dst_path = os.path.join(dst_set, class_name, img_name)
                copyfile(source_path, dst_path)
            
            print('[{}]/[{}]'.format(index_class, len(class_folders)))
            index_class = index_class + 1