import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import pandas as pd
import sys, os, glob, csv, json
import numpy as np
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

def splitTrainTestDataset(root_dir, out_dir):
    image_file_suffix = 'jpg'
    file_path_list = glob.glob(os.path.join(root_dir, '**', '**', '**', '*{}'.format(image_file_suffix)))

    dataset_size = len(file_path_list)
    indices = list(range(dataset_size))
    test_split = int(np.floor(0.2*dataset_size))
    val_split = int(np.floor(0.36*dataset_size))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[test_split:val_split]
    test_indices = indices[:test_split]

    train_path_list = []
    test_path_list = []
    val_path_list = []
    len_root_dir = len(root_dir)+1
    for idx in train_indices:
        train_path_list.append(file_path_list[idx][len_root_dir:])

    for idx in val_indices:
        val_path_list.append(file_path_list[idx][len_root_dir:])

    for idx in test_indices:
        test_path_list.append(file_path_list[idx][len_root_dir:])

    train_txt = os.path.join(out_dir, 'train.txt')
    val_txt = os.path.join(out_dir, 'val.txt')
    test_txt = os.path.join(out_dir, 'test.txt')

    with open(train_txt, 'w') as fp:
        for line in train_path_list:
            fp.writelines(line+'\n')

    with open(val_txt, 'w') as fp:
        for line in val_path_list:
            fp.writelines(line+'\n')
    
    with open(test_txt, 'w') as fp:
        for line in test_path_list:
            fp.writelines(line+'\n')

    return

class XDAData(Dataset):
    def __init__(self, image_path_file, root_dir,
                                transform=None):
        self.image_path_file = image_path_file
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._make_dataset()

    def _make_dataset(self):
        labels = {'plot': [], 'cultivar': [], 'scan_date': []}
        paths = []
        with open(self.image_path_file, 'r') as fp:
            in_paths = fp.readlines()

        for path in in_paths:
            path_split = path[:-1].split('/')
            plot = path_split[-2]
            if plot == 'protectWall':
                continue

            cultivar = plot.split('-')[0]
            scan_date = datetime.strptime(path_split[-4], '%Y-%m-%d_%H-%M-%S').date().toordinal()
            full_path = os.path.join(self.root_dir, path[:-1])
            paths.append(full_path)
            labels['plot'].append(plot)
            labels['cultivar'].append(int(cultivar))
            labels['scan_date'].append(scan_date)
            
        return paths, labels

    def get_labels(self):
        return self.labels

    def get_paths(self):
        return self.image_paths

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = {key: val[idx] for key, val in self.labels.items()}
        try:
            image = Image.open(image_path)
        except:
            print('Error Image: {}'.format(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label['cultivar']

    def __len__(self):
        return len(self.image_paths)
    
    def get_sample_label(self, sample):
        return sample[1]

class M1Data(Dataset):
    def __init__(self, image_path_file, label_file, root_dir,
                                transform=None):
        self.image_path_file = image_path_file
        self.label_file = label_file
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._make_dataset()

    def _make_dataset(self):
        labels = {'plot': [], 'cultivar': [], 'scan_date': []}
        paths = []
        with open(self.image_path_file, 'r') as fp:
            in_paths = fp.readlines()
        with open(self.label_file) as f:
            label_dict = json.load(f)


        for path in in_paths:
            path_split = path[:-1].split('/')
            plot = path_split[-2]
            cultivar = label_dict[plot]
            scan_date = path_split[-4]
            full_path = os.path.join(self.root_dir, path[:-1])
            paths.append(full_path)
            labels['plot'].append(plot)
            labels['cultivar'].append(int(cultivar))
            labels['scan_date'].append(scan_date)
            
        return paths, labels

    def get_labels(self):
        return self.labels

    def get_paths(self):
        return self.image_paths

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = {key: val[idx] for key, val in self.labels.items()}
        try:
            image = Image.open(image_path)
        except:
            print('Error Image: {}'.format(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label['cultivar']

    def __len__(self):
        return len(self.image_paths)
    
    def get_sample_label(self, sample):
        return sample[1]

def createNewCsv(in_file, out_file):

    label_dict = {}
    with open(in_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            s0 = '{}-{:03}'.format(row[0], int(row[2]))
            s1 = row[3]
            label_dict[s0] = s1

    with open(out_file, 'w') as f:
        json_file = json.dumps(label_dict)
        f.write(json_file)

    return

def test():
    root_dir = '/media/baker/C05A8B528B5B5A2D/data/baker/raw_data/M1/Z7II'
    label_file = '/home/baker/Work/data/raw_data/M1/excels/field_cultivar_phenptypes_less.csv'
    new_label_file = '/home/baker/Work/data/raw_data/M1/excels/field_cultivar_phenptypes.json'
    excel_dir = '/home/baker/Work/data/raw_data/M1/excels/'
    splitTrainTestDataset(root_dir, excel_dir)
    createNewCsv(label_file, new_label_file)

    test_file = os.path.join(excel_dir, 'test.txt')
    test_dataset = M1Data(test_file, new_label_file, root_dir)

    train_file = os.path.join(excel_dir, 'trainval.txt')
    train_dataset = M1Data(train_file, new_label_file, root_dir)

def createXDADataTxt():

    root_dir = '/media/baker/C05A8B528B5B5A2D/data/XDA24040201-1/Rice/data_2022/graph_data/Level1'
    out_dir = '/home/baker/Work/data/file_lists/XDA'
    #splitTrainTestDataset(root_dir, out_dir)

    test_file = os.path.join(out_dir, 'test.txt')
    test_dataset = XDAData(test_file, root_dir)

    train_file = os.path.join(out_dir, 'train.txt')
    train_dataset = XDAData(train_file, root_dir)
    

    return

createXDADataTxt()