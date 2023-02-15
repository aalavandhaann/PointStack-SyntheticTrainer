#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import json
import pathlib
import h5py
from glob import glob
import numpy as np

from .partnormal import PartNormal

from utils.runtime_utils import get_device

class SyntheticPartNormal(PartNormal):
    def __init__(self, cfg, class_choice=None, split='train', load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False):
        super().__init__(cfg = cfg, class_choice=None, split=split, load_name=True, load_file=True, random_rotate=False, random_jitter=False, random_translate=False)
        
        
        self.meta = {}

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.cat:
            self.meta[item] = []
            dir_point: pathlib.Path = pathlib.Path(os.path.join(self.root, self.cat[item]))
            # fns = sorted(os.listdir(dir_point))
            fns = sorted(list(dir_point.glob('**/*')))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn.stem in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn.stem in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn.stem in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn.stem in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = fn.stem#(os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))           
        
        self.seg_classes = {'Others': 0, 'Body-Parts': [i+1 for i in range(20)]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 2500
        self.device = get_device()

    def __len__(self):
        return len(self.datapath)

    '''
        The __getitem__ method for the synthetic data loader cannot have the cache concept.
        Reason is that shapenet datasets are just 3k points resulting in each file being under few 100 KB's
        However, the synthetic dataset contains point clouds that are minimum of 17MB. 
        Hence caching leads to RAM overflow and crashes. 
    '''
    def __getitem__(self, index):

        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)       

        # if self.normalize:
        #     point_set = pc_normalize(point_set)

        choice = np.random.choice(min(len(seg), self.npoints), self.npoints, replace=True)
        #choice = np.linspace(0, self.num_points, num=self.num_points).astype(int)
        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        if(self.cfg.GPU_COUNT > 1):
            data_dic = {
                'points'    : torch.from_numpy(point_set),
                'seg_id'    : torch.from_numpy(seg),
                'cls_tokens': torch.from_numpy(cls),
                'norms'     : torch.from_numpy(normal)
            }        
        else:
            data_dic = {
                'points'    : torch.from_numpy(point_set).to(self.device),#.cuda(),
                'seg_id'    : torch.from_numpy(seg).to(self.device),#.cuda(),
                'cls_tokens': torch.from_numpy(cls).to(self.device),#.cuda(),
                'norms'     : torch.from_numpy(normal).to(self.device)#.cuda()
            }        

        return data_dic