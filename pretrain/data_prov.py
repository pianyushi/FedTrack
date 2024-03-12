import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.sample_generator import SampleGenerator
from modules.utils import crop_image2


class RegionDataset(data.Dataset):  # 区域数据集
    def __init__(self, img_list, gt, opts):  # 图片存储路径 图片列表 gt
        self.img_list = np.asarray(img_list)
        self.gt = gt

        self.batch_frames = opts['batch_frames']  # 每个mini-batch是从一个视屏序列中随机取8帧图片
        self.batch_pos = opts['batch_pos']  # 32个正样本
        self.batch_neg = opts['batch_neg']  # 96个负样本

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']  # 图片尺寸107*107,不符合的话需要resize
        self.padding = opts['padding']

        self.flip = opts.get('flip', False)
        self.rotate = opts.get('rotate', 0)
        self.blur = opts.get('blur', 0)

        self.index = np.random.permutation(len(self.img_list))  # 生成一个打乱了顺序的索引数组
        self.pointer = 0  # 取的帧在index中的索引

        image = Image.open(self.img_list[0]).convert('RGB')  # 转换成RGB格式
        # 高斯分布生成一些正样本框,这里只执行了初始化步骤
        self.pos_generator = SampleGenerator('uniform', image.size,
                opts['trans_pos'], opts['scale_pos'])
        # 均匀分布生成一些负样本框,这里只执行了初始化步骤
        self.neg_generator = SampleGenerator('uniform', image.size,
                opts['trans_neg'], opts['scale_neg'])

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]  # 每次取8帧
        if len(idx) < self.batch_frames:  # 不够的重新取8帧
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer  # 更新下一次取得8帧
        # 生成两个空数组 array([], shape=(0, 3, 107, 107), dtype=float64)
        pos_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):  # 随机抽8帧，然后找到32个正样本和96个负样本
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)  # 将RGB图像转换为数组 (360, 480, 3)

            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)  # 下取整,n_pos = 4,4,4,4,4,4,4,4
            n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i)  # n_neg = 12,12,12,12,12,12,12,12
            pos_examples = self.pos_generator(bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = self.neg_generator(bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)), axis=0)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)), axis=0)

        pos_regions = torch.from_numpy(pos_regions)
        neg_regions = torch.from_numpy(neg_regions)
        return pos_regions, neg_regions  # 返回32个正样本区域和96个负样本区域，构成了每次训练的mini-batch

    next = __next__

    def extract_regions(self, image, samples):  # 从image中选出samples的区域
        # regions 即是len(samples)个107*107*3RGB的图像
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image2(image, sample, self.crop_size, self.padding,
                    self.flip, self.rotate, self.blur)
        regions = regions.transpose(0, 3, 1, 2)  # 改变维度顺序，相当于把107*107*3变成了3*107*107 ？？？为什么要变呢？
        regions = regions.astype('float32') - 128.
        return regions
