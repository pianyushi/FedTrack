import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.utils import crop_image2


class RegionExtractor():  # 区域提取
    def __init__(self, image, samples, opts):
        self.image = np.asarray(image)
        self.samples = samples

        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.batch_size = opts['batch_test']

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):  # 采集完了所有的样本区域
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index)  # 根据samples提取样本区域
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):  # 根据samples从图像中提取样本区域
            regions[i] = crop_image2(self.image, sample, self.crop_size, self.padding)
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.  # 数据的0中心化
        return regions
