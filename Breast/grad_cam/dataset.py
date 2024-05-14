
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

        LOW_ENERGY = h5f['LOW_ENERGY'][:].astype(np.float32)
        HIGH_ENERGY = h5f['HIGH_ENERGY'][:].astype(np.float32)
        ENHANCE = h5f['ENHANCE'][:].astype(np.float32)
        label = h5f['label'][()]


        LOW_ENERGY = cv2.resize(LOW_ENERGY, (320, 320))
        HIGH_ENERGY = cv2.resize(HIGH_ENERGY, (320, 320))
        ENHANCE = cv2.resize(ENHANCE, (320, 320))



        seed = np.random.randint(255)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY = Image.fromarray(np.uint8(LOW_ENERGY*255))
            torch.manual_seed(seed)
            HIGH_ENERGY= Image.fromarray(np.uint8(HIGH_ENERGY * 255))
            torch.manual_seed(seed)
            ENHANCE = Image.fromarray(np.uint8(255 * ENHANCE))
            #
            torch.manual_seed(seed)
            LOW_ENERGY = self.transform(LOW_ENERGY)

            torch.manual_seed(seed)
            HIGH_ENERGY=self.transform(HIGH_ENERGY)

            torch.manual_seed(seed)
            ENHANCE = self.transform(ENHANCE)


        sample = {'LOW_ENERGY': LOW_ENERGY, 'HIGH_ENERGY': HIGH_ENERGY,'ENHANCE': ENHANCE,
                  'label': label} #三输入融合
        
        # sample = {'LOW_ENERGY': LOW_ENERGY, 'HIGH_ENERGY': LOW_ENERGY,'ENHANCE': LOW_ENERGY,
        #           'label': label} # 都是T1
        
        # sample = {'LOW_ENERGY': HIGH_ENERGY, 'HIGH_ENERGY': HIGH_ENERGY,'ENHANCE': HIGH_ENERGY,
        #           'label': label} # 都是T2
        
        # sample = {'LOW_ENERGY': ENHANCE, 'HIGH_ENERGY': ENHANCE,'ENHANCE': ENHANCE,
        #           'label': label} # 都是enhance
        
        # sample = {'LOW_ENERGY': LOW_ENERGY, 'HIGH_ENERGY': HIGH_ENERGY,'ENHANCE': HIGH_ENERGY,
        #           'label': label} # T1和T2融合
        
        # sample = {'LOW_ENERGY': LOW_ENERGY, 'HIGH_ENERGY': LOW_ENERGY,'ENHANCE': ENHANCE,
        #           'label': label} # T1和enhance融合
        
        # sample = {'LOW_ENERGY': HIGH_ENERGY, 'HIGH_ENERGY': HIGH_ENERGY,'ENHANCE': ENHANCE,
        #           'label': label} # T2和enhance融合

        return sample
