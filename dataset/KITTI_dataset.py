import sys
sys.path.append('..')
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from path import Path
from utils.utils import read_pose_from_text  # Assuming this function reads the 3x4 extrinsic matrix
from utils import custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

IMU_FREQ = 10

class KITTI(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=['00', '01', '02', '04', '05','06'],
                 transform=None):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:
            gts = np.loadtxt(self.root/'poses/{}.txt'.format(folder))  # Load 3x4 extrinsic matrix
            imus = sio.loadmat(self.root/'imus/{}.mat'.format(folder))['imu_data_interp']
            
            fpaths = sorted((self.root/'sequences/{}/image_2'.format(folder)).files("*.png"))      
            for i in range(len(fpaths)-self.sequence_length):
                img_samples = fpaths[i:i+self.sequence_length]
                imu_samples = imus[i*IMU_FREQ:(i+self.sequence_length-1)*IMU_FREQ+1]
                extrinsic_samples = gts.reshape(3,4)
                sample = {'imgs': img_samples, 'imus': imu_samples, 'gts': extrinsic_samples}
                sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.asarray(Image.open(img)) for img in sample['imgs']]
        
        if self.transform is not None:
            imgs, imus, gts = self.transform(imgs, np.copy(sample['imus']), sample['gts'])
        else:
            imus = np.copy(sample['imus'])
            gts = sample['gts']
        
        return imgs, imus, gts

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Training sequences: '
        for seq in self.train_seqs:
            fmt_str += '{} '.format(seq)
        fmt_str += '\n'
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

# Create a DataLoader to load batches of the KITTI dataset
dataset = KITTI(root='/mnt/ddb5152f-7808-4a33-9f27-ba0a0d7f3164/Naitri/AutoCalib/visual-selective-vio/data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
