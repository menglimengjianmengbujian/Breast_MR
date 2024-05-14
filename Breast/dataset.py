
import os
import sys
import pickle

import matplotlib.pyplot
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

class CESM(Dataset):
    def __init__(self, base_dir=None, labeled_num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.targets = []
        # self.split = split
        self.transform = transform
        dir = os.listdir(self._base_dir)
        dir.sort()
        for name in dir:
            image = os.path.join(self._base_dir, name)
            self.sample_list.append(image)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')

        data_DCE = h5f['data_DCE'][:].astype(np.float32)
        data_T2 = h5f['data_T2'][:].astype(np.float32)
        csv = h5f['csv'][:].astype(np.float32)

        label = h5f['label'][()]


        # data_DCE = cv2.resize(data_DCE, (320, 320))
        # data_T2 = cv2.resize(data_T2, (320, 320))
        # csv = cv2.resize(csv , (320, 320))



        seed = np.random.randint(255)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            data_DCE = Image.fromarray(np.uint8(data_DCE.transpose(1, 2 , 0 )*255))
            torch.manual_seed(seed)
            data_T2= Image.fromarray(np.uint8(data_T2.transpose(1, 2 , 0 )* 255))

            # torch.manual_seed(seed)
            # ENHANCE = Image.fromarray(np.uint8(255 * csv))

            torch.manual_seed(seed)
            data_DCE = self.transform(data_DCE)

            torch.manual_seed(seed)
            data_T2=self.transform(data_T2)

            torch.manual_seed(seed)
            csv = csv


        # sample = {'data_DCE': data_DCE, 'data_T2': data_T2,'csv': csv,
        #           'label': label} #三输入融合

        # sample = {'data_DCE': data_DCE, 'data_T2': data_DCE,'csv': 'csv': csv
        #           'label': label} # 都是DCE

        
        sample = {'data_DCE': data_T2, 'data_T2': data_T2,'csv': csv,
                  'label': label} #都是T2
        


        return sample
